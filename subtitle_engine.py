"""
Subgen - 智能字幕生成引擎
支持 Whisper + FunASR 双引擎
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
import json
import datetime
import ffmpeg

import whisper
import argostranslate.package
import argostranslate.translate
from tqdm import tqdm
try:
    from funasr import AutoModel
except ImportError:
    pass  # Handle import error gracefully if needed, or rely on runtime check

from model_manager import ModelManager


class SubtitleEngine:
    """字幕生成引擎"""
    
    def __init__(self):
        """初始化引擎"""
        self.model_manager = ModelManager()
        self.loaded_models = {}
        self.user_dict = self._load_user_dict()
        self._first_run_check()
    
    def _first_run_check(self):
        """首次运行检查（仅日志记录，GUI 会处理安装）"""
        status = self.model_manager.check_installation()
        required_models = [k for k, v in ModelManager.MODELS.items() if v['required']]
        missing = [k for k in required_models if not status.get(k, False)]
        
        if missing:
            print(f"ℹ️  检测到缺少必需模型: {', '.join(missing)}")
            # GUI will handle installation prompting
    
    def _load_user_dict(self) -> Dict[str, str]:
        """加载用户词典"""
        dict_path = Path.home() / '.subgen' / 'user_dict.json'
        
        if dict_path.exists():
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        
        default_dict = {
            'kopitiam': '咖啡店',
            'kopi': '咖啡',
            'teh': '茶',
            'makan': '吃饭',
            'laksa': '叻沙',
        }
        
        try:
            dict_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dict_path, 'w', encoding='utf-8') as f:
                json.dump(default_dict, f, ensure_ascii=False, indent=2)
        except Exception:
            pass # Fail silently if cannot write
        
        return default_dict
    
    def load_model(self, model_key: str):
        """加载模型"""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        model_info = ModelManager.MODELS[model_key]
        print(f"⏳ 加载 {model_info['name']}...")
        
        if model_info['provider'] == 'openai':
            # Whisper handles caching internally via download_root
            download_root = str(self.model_manager.cache_dir / 'whisper')
            # Verify if file exists first to avoid re-download attempt if network is down
            # model_path = Path(download_root) / f"{model_info['model_id']}.pt"
            
            model = whisper.load_model(model_info['model_id'], download_root=download_root)
            self.loaded_models[model_key] = {
                'type': 'whisper',
                'model': model,
                'info': model_info
            }
        elif model_info['provider'] == 'modelscope':
            # FunASR - specify cache to avoid "Downloading" message
            cache_dir = self.model_manager.cache_dir / 'modelscope'
            
            model = AutoModel(
                model=model_info['model_id'], 
                trust_remote_code=True,
                disable_update=True,
                cache_dir=str(cache_dir)
            )
            self.loaded_models[model_key] = {
                'type': 'funasr',
                'model': model,
                'info': model_info
            }
        
        print(f"✅ 模型加载完成\n")
        return self.loaded_models[model_key]
    
    def transcribe(
        self,
        video_path: str,
        model_key: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict:
        """智能转录"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"文件不存在: {video_path}")
        
        print(f"🎬 开始处理: {Path(video_path).name}\n")
        
        if model_key is None:
            model_key = self._auto_select_model(video_path, language)
        
        print(f"🤖 使用模型: {ModelManager.MODELS[model_key]['name']}")
        
        # 简单的长视频检测逻辑 (超过 100MB 或 10分钟假设)
        # 为了准确我们读取 metadata，这里简化：如果是 Whisper 模型且文件较大，使用分段处理
        # 或者用户强制开启？
        # 目前策略：默认尝试直接处理，但如果文件 > 200MB 且是 Whisper，尝试分段
        # (因为 FunASR 通常处理较好，而 Whisper 长时间容易漂移)
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > 200 and 'whisper' in model_key:
            try:
                # 尝试使用长视频模式
                return self.transcribe_long_video(video_path, model_key, language)
            except Exception as e:
                print(f"⚠️ 长视频模式失败，回退到普通模式: {e}")
        
        # 加载并转录
        
        # 加载并转录
        model_data = self.load_model(model_key)
        
        result = {}
        if model_data['type'] == 'whisper':
            result = self._transcribe_whisper(model_data['model'], video_path, language)
        elif model_data['type'] == 'funasr':
            result = self._transcribe_funasr(model_data['model'], video_path)
        
        # DEBUG LOG
        print(f"    [DEBUG] Raw Segments Count: {len(result['segments'])}")
        if result['segments']:
            print(f"    [DEBUG] First Segment: {result['segments'][0]}")
        else:
            print(f"    [DEBUG] No segments found. Raw text: {result.get('text', '')[:100]}...")
            if 'whisper' in model_key:
                print("    [DEBUG] Hint: If using Whisper, ensure ffmpeg is installed and working.")
        
        # 应用用户词典
        result = self._apply_user_dict(result)
        
        print(f"✅ 识别完成！共 {len(result['segments'])} 个片段\n")
        return result
    
    def transcribe_long_video(self, video_path, model_key, language):
        """
        处理长视频：切分为 10 分钟片段防止时间轴漂移
        """
        print("🔄 检测到长视频模式，正在进行分段处理...")
        from pydub import AudioSegment
        import math
        import shutil
        
        # 1. 转换/加载音频
        # Pydub can load directly from mp4 if ffmpeg is available
        try:
            audio = AudioSegment.from_file(video_path)
        except Exception as e:
            print(f"⚠️ 无法加载音频，将尝试直接转录: {e}")
            model_data = self.load_model(model_key)
            if model_data['type'] == 'whisper':
                return self._transcribe_whisper(model_data['model'], video_path, language)
            return self._transcribe_funasr(model_data['model'], video_path)

        # 2. 切分 (10分钟 = 600,000ms)
        chunk_len = 10 * 60 * 1000
        total_len = len(audio)
        chunks = math.ceil(total_len / chunk_len)
        
        combined_segments = []
        temp_dir = Path(video_path).parent / "subgen_temp"
        temp_dir.mkdir(exist_ok=True)
        
        model_data = self.load_model(model_key)
        
        print(f"📦 将视频分为 {chunks} 个片段进行处理...")
        
        try:
            for i in range(chunks):
                start_ms = i * chunk_len
                end_ms = min((i + 1) * chunk_len, total_len)
                
                # 导出片段
                chunk = audio[start_ms:end_ms]
                chunk_path = temp_dir / f"chunk_{i}.wav"
                chunk.export(chunk_path, format="wav")
                
                print(f"   ► 处理片段 {i+1}/{chunks}...")
                
                # 转录片段
                if model_data['type'] == 'whisper':
                    res = self._transcribe_whisper(model_data['model'], str(chunk_path), language)
                else:
                    res = self._transcribe_funasr(model_data['model'], str(chunk_path))
                
                # 调整时间轴并合并
                offset_seconds = start_ms / 1000.0
                for seg in res['segments']:
                    seg['start'] += offset_seconds
                    seg['end'] += offset_seconds
                    combined_segments.append(seg)
                    
        finally:
            # 清理临时文件
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
                
        return {
            'language': language or 'known',
            'text': " ".join([s['text'] for s in combined_segments]),
            'segments': combined_segments
        }
    
    def _auto_select_model(self, video_path: str, language: Optional[str]) -> str:
        """自动根据语言选择模型"""
        # 中文优先使用 Whisper（完整度更高）
        # FunASR 虽然方言好但 VAD 太严格导致识别不完整
        # 用户可以手动指定 --model funasr 强制使用
        if language == 'zh' or language == 'zh-CN':
            print("    ℹ️  中文检测：使用 Whisper（完整度优先）")
            print("    💡 如需更好的方言识别，请手动指定: --model funasr")
            return 'whisper_medium'
        if language == 'en':
            return 'whisper_medium'
            
        # 如果未指定语言，尝试使用 whisper tiny 快速检测
        if language is None:
            print("🕵️ 正在检测语言...")
            try:
                model = whisper.load_model("tiny")
                audio = whisper.load_audio(video_path)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(model.device)
                _, probs = model.detect_language(mel)
                detected_lang = max(probs, key=probs.get)
                print(f"    检测结果: {detected_lang}")
                
                # 中文也用 Whisper
                return 'whisper_medium'
            except Exception as e:
                print(f"    语言检测失败: {e}，默认使用 Whisper Medium")
                return 'whisper_medium'
        
        return 'whisper_medium'

    def _transcribe_whisper(self, model, video_path: str, language: Optional[str]) -> Dict:
        """使用 Whisper 进行转录"""
        # language override if provided
        options = {}
        if language:
            options['language'] = language
            
        # Transcribe
        result = model.transcribe(video_path, **options)
        
        # Standardize segments
        segments = []
        for seg in result['segments']:
            segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip()
            })
            
        return {
            'language': result.get('language', 'unknown'),
            'text': result['text'],
            'segments': segments
        }

    def _transcribe_funasr(self, model, video_path: str) -> Dict:
        """使用 FunASR 进行转录"""
        try:
            # Generate with optimal parameters + relaxed VAD for better coverage
            hotwords = ' '.join(self.user_dict.keys()) if self.user_dict else ''
            
            res = model.generate(
                input=video_path,
                hotword=hotwords,
                batch_size_s=0,  # Process entire file
                # VAD parameters to capture more speech (especially dialects/low volume)
                vad_model="fsmn-vad",  # Use built-in VAD
                vad_kwargs={
                    "max_single_segment_time": 60000,  # Max 60s per segment
                    "speech_noise_thres": 0.6,  # Lower threshold = more sensitive (default ~0.8)
                },
                # Disable overly aggressive silence filtering
                merge_vad=True,  # Merge close segments
                merge_length_s=15  # Merge segments within 15s
            )
        except Exception as e:
             raise RuntimeError(f"FunASR execution failed: {e}")
        
        if not res:
            return {'language': 'zh', 'text': '', 'segments': []}
            
        item = res[0]
        
        # --- DEBUG START ---
        print(f"    [DEBUG] FunASR Item Keys: {item.keys()}")
        if 'text' in item:
             print(f"    [DEBUG] Text Snippet: {item['text'][:50]}...")
        # --- DEBUG END ---

        full_text = item.get('text', '')
        segments = []
        
        # 1. Try 'sentence_info' (Best case)
        if 'sentence_info' in item:
            for sent in item['sentence_info']:
                # Ensure structure is what we expect
                if 'timestamp' in sent:
                    # timestamp might be [[start, end]] or just [start, end]
                    # usually sentence_info timestamp is often simple [start, end] or similar?
                    # Let's check structure dynamically if possible or assume standard
                    ts = sent['timestamp']
                    # logic to handle ts format if needed
                
                # Default mapping
                # sent: {'text': '...', 'start': ms, 'end': ms, ...} usually in updated version
                # If 'start' not in sent, maybe in 'timestamp'
                
                start_sec = 0
                end_sec = 0
                
                if 'start' in sent and 'end' in sent:
                    start_sec = sent['start'] / 1000.0
                    end_sec = sent['end'] / 1000.0
                elif 'timestamp' in sent:
                    # timestamp: [[start, end]]
                    if isinstance(sent['timestamp'], list) and len(sent['timestamp']) > 0:
                         if isinstance(sent['timestamp'][0], list):
                             start_sec = sent['timestamp'][0][0] / 1000.0
                             end_sec = sent['timestamp'][-1][1] / 1000.0
                         else:
                             start_sec = sent['timestamp'][0] / 1000.0
                             end_sec = sent['timestamp'][1] / 1000.0
                
                segments.append({
                    'start': start_sec,
                    'end': end_sec,
                    'text': sent['text']
                })

        # 2. If no sentence_info, check raw 'timestamp' (Token level)
        elif 'timestamp' in item and 'text' in item:
             print("    [DEBUG] Using fallback segmentation from token timestamps.")
             
             # text is space-separated tokens: "玩 诈 骗 我"
             # timestamp is [[s,e], [s,e], ...] one per token in ms
             
             tokens = item['text'].split()
             timestamps = item['timestamp']
             
             print(f"    [DEBUG] Total tokens to process: {len(tokens)}")
             
             if len(tokens) != len(timestamps):
                 print(f"    [WARN] Token count {len(tokens)} != Timestamp count {len(timestamps)}.")
             
             # Group tokens into subtitle segments
             current_seg_text = []
             current_start = None
             last_end = 0
             char_count = 0
             
             # Punctuation marks that should trigger segment breaks
             punctuation = {'。', '！', '？', '，', '、', '.', '!', '?', ','}
             
             limit = min(len(tokens), len(timestamps))
             
             for i in range(limit):
                 token = tokens[i]
                 ts = timestamps[i]  # [start_ms, end_ms]
                 
                 if not isinstance(ts, list) or len(ts) < 2:
                     continue
                     
                 start_ms = ts[0]
                 end_ms = ts[1]
                 
                 # Add token first
                 if current_start is None:
                     current_start = start_ms
                 
                 current_seg_text.append(token)
                 char_count += len(token)
                 
                 # Split logic:
                 # 1. Hit punctuation (except comma if too short)
                 # 2. Silence gap > 800ms
                 # 3. Accumulated > 15 chars
                 gap = (start_ms - last_end) if last_end > 0 else 0
                 is_gap = gap > 800
                 is_long = char_count >= 15
                 has_punct = token in punctuation
                 
                 # Force break on period/question/exclamation or if too long
                 should_break = (
                     (has_punct and token in {'。', '！', '？', '.', '!', '?'}) or
                     (is_gap and char_count > 3) or
                     is_long
                 )
                 
                 if should_break and current_seg_text:
                     # Flush current segment
                     segments.append({
                         'start': current_start / 1000.0,
                         'end': end_ms / 1000.0,
                         'text': "".join(current_seg_text)
                     })
                     current_seg_text = []
                     current_start = None
                     char_count = 0
                 
                 last_end = end_ms
             
             # Flush remaining
             if current_seg_text:
                 segments.append({
                     'start': current_start / 1000.0,
                     'end': last_end / 1000.0,
                     'text': "".join(current_seg_text)
                 })
             
             print(f"    [DEBUG] Generated {len(segments)} segments from {limit} tokens.")
             
        return {
            'language': 'zh',
            'text': full_text,
            'segments': segments
        }

    def _apply_user_dict(self, result: Dict) -> Dict:
        """应用用户词典替换文本"""
        if not self.user_dict:
            return result
            
        for segment in result['segments']:
            text = segment['text']
            for wrong, right in self.user_dict.items():
                if wrong in text: # Simple replacement, case sensitive for now?
                    text = text.replace(wrong, right)
                # Case insensitive check could be better for english keys
                elif wrong.lower() in text.lower():
                     # This is a bit complex to preserve case of surrounding text, 
                     # but for MVP plain replace is okay.
                     import re
                     pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                     text = pattern.sub(right, text)
            segment['text'] = text
            
        return result

    def _get_video_fps(self, video_path: str) -> float:
        """获取视频帧率"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream:
                r_frame_rate = video_stream['r_frame_rate']
                num, den = map(int, r_frame_rate.split('/'))
                return num / den
        except Exception as e:
            print(f"    ⚠️ 无法获取视频帧率: {e}，将使用默认时间轴")
        return 0.0

    def save_srt(self, result: Dict, output_path: str, video_path: Optional[str] = None) -> str:
        """保存为 SRT 文件（自动版本号避免覆盖 + 帧对齐）"""
        # Check if file exists and auto-version if needed
        if os.path.exists(output_path):
            base_path = Path(output_path)
            stem = base_path.stem
            parent = base_path.parent
            counter = 1
            
            while os.path.exists(output_path):
                output_path = str(parent / f"{stem}_{counter}.srt")
                counter += 1
            
            print(f"    ℹ️  文件已存在，保存为: {Path(output_path).name}")
        
        # Get video FPS for frame alignment
        fps = 0.0
        if video_path:
            fps = self._get_video_fps(video_path)
            if fps > 0:
                print(f"    🎬 视频帧率: {fps:.2f} fps - 正在执行帧对齐...")

        segments = result['segments']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                start = self._format_timestamp(seg['start'], fps)
                end = self._format_timestamp(seg['end'], fps)
                text = seg['text']
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        
        return output_path

    def _format_timestamp(self, seconds: float, fps: float = 0.0) -> str:
        """格式化时间戳，可选帧对齐"""
        if seconds is None:
            return "00:00:00,000"
            
        seconds = float(seconds)
        
        # Frame alignment logic
        if fps > 0:
            frame_duration = 1.0 / fps
            # Snap to nearest frame index
            frame_index = round(seconds * fps)
            # Re-calculate seconds based on frame index
            seconds = frame_index * frame_duration
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
        
    def translate_srt(self, srt_path: str, target_lang: str) -> str:
        """
        Translates an SRT file to the target language.
        Generates a pure translated SRT file (e.g. .zh.srt).
        """
        import pysrt
        
        if target_lang == 'zh':
            from_code = 'en'
            to_code = 'zh'
        else: # target 'en'
            from_code = 'zh'
            to_code = 'en'
            
        print(f"🎬 准备翻译: {from_code} -> {to_code}")
        
        # Configure Argos to use our cache directory
        # Set env var before importing/using argostranslate features if possible
        # or use internal API if available. Argos uses XDG_DATA_HOME or ~/.local by default.
        import os
        argos_dir = self.model_manager.cache_dir / 'argos'
        argos_dir.mkdir(parents=True, exist_ok=True)
        os.environ['ARGOS_PACKAGES_DIR'] = str(argos_dir)
        
        # 1. Check and Install Model (Same logic)
        try:
            # Re-import to ensure env var is picked up if it's checked at module level
            # Actually argostranslate.package uses settings that might be cached.
            # Ideally we set this in __init__? For now setting here and hoping package.* uses it dynamically
            # or we might need to look at how argostranslate initializes.
            # A safer bet is setting package.data_dir if exposed.
            
            # According to docs/source, argostranslate uses XDG_DATA_HOME/argos-translate/packages
            # We will rely on ARGOS_PACKAGES_DIR which is supported in newer versions specific overrides.
            
            argostranslate.package.update_package_index()
            
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                filter(
                    lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
                ), None
            )
            
            if package_to_install:
                installed_packages = argostranslate.package.get_installed_packages()
                is_installed = any(
                    p.from_code == from_code and p.to_code == to_code for p in installed_packages
                )
                
                if not is_installed:
                    print(f"📥 下载翻译模型 ({from_code}->{to_code})...")
                    argostranslate.package.install_from_path(package_to_install.download())
            else:
                print(f"⚠️ 未找到翻译模型: {from_code} -> {to_code}")
                # For MVP, continue and maybe fail later or just copy
        except Exception as e:
            print(f"⚠️ 模型检查失败: {e}")

        # 2. Translate content
        print("🔄 正在翻译字幕...")
        subs = pysrt.open(srt_path)
        new_subs = pysrt.SubRipFile()
        
        for sub in tqdm(subs, desc="Translating"):
            original_text = sub.text.replace('\n', ' ')
            
            try:
                translated_text = argostranslate.translate.translate(original_text, from_code, to_code)
            except Exception as e:
                translated_text = "[Translation Failed]"
            
            # Save ONLY translated text here
            item = pysrt.SubRipItem(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                text=translated_text
            )
            new_subs.append(item)
            
        # Save as separate language file
        new_path = str(Path(srt_path).with_suffix(f'.{to_code}.srt'))
        new_subs.save(new_path, encoding='utf-8')
        print(f"✨ 译文已保存: {new_path}")
        
        return new_path

    def merge_subtitles(self, original_path: str, translated_path: str, track_order: List[str], output_path: str):
        """
        Merge two SRT files based on track order with intelligent timecode handling and alignment.
        track_order: ['original', 'translated'] or ['translated', 'original']
        
        Auto-conversion:
        - Detects non-standard timecode formats (HH:MM:SS:FF)
        - Automatically converts to standard SRT format (HH:MM:SS,mmm)
        - Seamless one-click operation
        
        Timecode Strategy:
        - If timecodes match closely (within 100ms), use averaged timecode
        - Otherwise, use the timecode from the longer/more complete file
        - Handles files with different lengths gracefully
        
        Intelligent Alignment:
        - Detects when one language has split sentences while the other doesn't
        - Automatically aligns based on start/end timecodes
        - Duplicates text from fewer subtitles to match more subtitles
        """
        import pysrt
        import re
        import tempfile
        
        # Auto-convert non-standard formats if needed
        original_path = self._auto_convert_srt_format(original_path)
        translated_path = self._auto_convert_srt_format(translated_path)
        
        try:
            subs_orig = pysrt.open(original_path, encoding='utf-8')
            subs_trans = pysrt.open(translated_path, encoding='utf-8')
        except Exception as e:
            raise ValueError(f"无法读取SRT文件: {e}")
        
        # Check for empty files with detailed error message
        if len(subs_orig) == 0 and len(subs_trans) == 0:
            raise ValueError(f"两个SRT文件都是空的！\n文件1: {Path(original_path).name}\n文件2: {Path(translated_path).name}")
        elif len(subs_orig) == 0:
            raise ValueError(f"文件1是空的（没有字幕）：{Path(original_path).name}\n请检查文件内容")
        elif len(subs_trans) == 0:
            raise ValueError(f"文件2是空的（没有字幕）：{Path(translated_path).name}\n请检查文件内容")
        
        print(f"    📊 文件1: {len(subs_orig)} 条字幕")
        print(f"    📊 文件2: {len(subs_trans)} 条字幕")
        
        # Check if counts match
        if len(subs_orig) == len(subs_trans):
            print(f"    ✓ 字幕数量匹配，使用简单合并模式")
            return self._merge_aligned_subtitles(subs_orig, subs_trans, track_order, output_path, original_path, translated_path)
        else:
            print(f"    ⚠️  字幕数量不匹配，启用智能对齐模式")
            return self._merge_with_alignment(subs_orig, subs_trans, track_order, output_path, original_path, translated_path)
    
    def _auto_convert_srt_format(self, srt_path: str) -> str:
        """
        Auto-detect and convert non-standard SRT timecode formats.
        Returns the path to a valid SRT file (original or converted temp file).
        """
        import re
        import tempfile
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file uses non-standard timecode format (HH:MM:SS:FF - HH:MM:SS:FF)
            # Standard SRT format: HH:MM:SS,mmm --> HH:MM:SS,mmm
            timecode_pattern = r'\d{2}:\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}:\d{2}'
            
            if re.search(timecode_pattern, content):
                print(f"    🔄 检测到非标准格式，自动转换中...")
                
                # Convert the format
                converted_path = self._convert_timecode_format(srt_path)
                print(f"    ✓ 格式转换完成")
                return converted_path
            else:
                # Already in standard format
                return srt_path
                
        except Exception as e:
            print(f"    ⚠️  格式检测失败，使用原文件: {e}")
            return srt_path
    
    def _convert_timecode_format(self, input_path: str, fps: int = 25) -> str:
        """
        Convert timecode format from HH:MM:SS:FF to HH:MM:SS,mmm
        Returns path to converted temp file.
        """
        import re
        import tempfile
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        output_lines = []
        subtitle_index = 1
        i = 0
        
        # Timecode pattern: HH:MM:SS:FF - HH:MM:SS:FF
        timecode_pattern = r'(\d{2}):(\d{2}):(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2}):(\d{2}):(\d{2})'
        
        while i < len(lines):
            line = lines[i].strip()
            match = re.match(timecode_pattern, line)
            
            if match:
                # Extract time components
                h1, m1, s1, f1 = match.groups()[:4]
                h2, m2, s2, f2 = match.groups()[4:]
                
                # Convert frames to milliseconds
                ms1 = int(f1) * 1000 // fps
                ms2 = int(f2) * 1000 // fps
                
                # Build standard SRT timecode
                start_time = f"{h1}:{m1}:{s1},{ms1:03d}"
                end_time = f"{h2}:{m2}:{s2},{ms2:03d}"
                
                # Write subtitle index
                output_lines.append(f"{subtitle_index}\n")
                
                # Write timecode
                output_lines.append(f"{start_time} --> {end_time}\n")
                
                # Collect subtitle text (until empty line)
                i += 1
                while i < len(lines) and lines[i].strip():
                    output_lines.append(lines[i])
                    i += 1
                
                # Add separator
                output_lines.append("\n")
                subtitle_index += 1
            
            i += 1
        
        # Save to temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.srt', text=True)
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                f.writelines(output_lines)
        except:
            os.close(temp_fd)
            raise
        
        return temp_path
    
    def _merge_aligned_subtitles(self, subs_orig, subs_trans, track_order, output_path, original_path, translated_path):
        """Simple merge when subtitle counts match"""
        import pysrt
        
        merged_subs = pysrt.SubRipFile()
        
        for i in range(len(subs_orig)):
            orig_item = subs_orig[i]
            trans_item = subs_trans[i]
            
            # Average timecodes if they're close
            start_diff = abs(orig_item.start.ordinal - trans_item.start.ordinal)
            end_diff = abs(orig_item.end.ordinal - trans_item.end.ordinal)
            
            if start_diff < 100 and end_diff < 100:
                avg_start = (orig_item.start.ordinal + trans_item.start.ordinal) // 2
                avg_end = (orig_item.end.ordinal + trans_item.end.ordinal) // 2
                final_start = pysrt.SubRipTime(milliseconds=avg_start)
                final_end = pysrt.SubRipTime(milliseconds=avg_end)
            else:
                final_start = orig_item.start
                final_end = orig_item.end
            
            # Combine texts
            parts = []
            for track in track_order:
                if track == 'original':
                    parts.append(orig_item.text)
                elif track == 'translated':
                    parts.append(trans_item.text)
            
            combined_text = "\n".join(parts)
            
            item = pysrt.SubRipItem(
                index=i + 1,
                start=final_start,
                end=final_end,
                text=combined_text
            )
            merged_subs.append(item)
        
        merged_subs.save(output_path, encoding='utf-8')
        print(f"✨ 合并字幕已保存: {output_path}")
        print(f"   共 {len(merged_subs)} 条字幕")
        return output_path
    
    def _merge_with_alignment(self, subs_orig, subs_trans, track_order, output_path, original_path, translated_path):
        """
        Intelligent merge when subtitle counts don't match.
        Uses the file with MORE subtitles as base and duplicates text from the file with FEWER subtitles.
        
        Example: 
        - Chinese: 3 sentences (split)
        - English: 1 sentence (complete)
        → Output: 3 subtitles, each with English text duplicated
        """
        import pysrt
        
        # Determine which has more subtitles - THIS IS THE BASE
        if len(subs_orig) > len(subs_trans):
            more_subs = subs_orig  # Base with more entries
            less_subs = subs_trans  # Source to duplicate from
            more_is_orig = True
        else:
            more_subs = subs_trans  # Base with more entries
            less_subs = subs_orig  # Source to duplicate from
            more_is_orig = False
        
        print(f"    🔍 对齐策略: 使用 {len(more_subs)} 条作为基准，复制 {len(less_subs)} 条的文本")
        
        merged_subs = pysrt.SubRipFile()
        less_idx = 0
        
        for more_idx in range(len(more_subs)):
            more_item = more_subs[more_idx]
            
            # Find the corresponding less_item that covers this more_item's time
            # Look for the less_item whose time range contains or overlaps with more_item
            matched_less_item = None
            
            # Search from current less_idx position
            for search_idx in range(less_idx, min(less_idx + 10, len(less_subs))):
                less_item = less_subs[search_idx]
                
                # Check if less_item's time range covers or overlaps with more_item
                # Overlap if: less_start <= more_end AND less_end >= more_start
                overlaps = (less_item.start.ordinal <= more_item.end.ordinal and
                           less_item.end.ordinal >= more_item.start.ordinal)
                
                if overlaps:
                    matched_less_item = less_item
                    
                    # Update less_idx for next iteration if we're getting close to the end
                    # of less_item's time range
                    if more_item.end.ordinal >= less_item.end.ordinal - 200:
                        less_idx = min(search_idx + 1, len(less_subs) - 1)
                    
                    break
            
            # If no match found, use the current less_idx item or previous one
            if not matched_less_item:
                if less_idx < len(less_subs):
                    matched_less_item = less_subs[less_idx]
                else:
                    matched_less_item = less_subs[-1]  # Use last item
            
            # Use more_item's timecode (the split one)
            final_start = more_item.start
            final_end = more_item.end
            
            # Get texts
            more_text = more_item.text
            less_text = matched_less_item.text
            
            # Determine which is original and which is translated
            if more_is_orig:
                text_orig = more_text
                text_trans = less_text  # This will be duplicated across splits
            else:
                text_orig = less_text  # This will be duplicated across splits
                text_trans = more_text
            
            # Combine based on track order
            parts = []
            for track in track_order:
                if track == 'original':
                    parts.append(text_orig)
                elif track == 'translated':
                    parts.append(text_trans)
            
            combined_text = "\n".join(parts)
            
            item = pysrt.SubRipItem(
                index=more_idx + 1,
                start=final_start,
                end=final_end,
                text=combined_text
            )
            merged_subs.append(item)
        
        # Save result
        merged_subs.save(output_path, encoding='utf-8')
        print(f"✨ 合并字幕已保存: {output_path}")
        print(f"   共 {len(merged_subs)} 条字幕")
        print(f"   📝 从 {len(less_subs)} 条复制文本到 {len(more_subs)} 条分割片段")
        return output_path