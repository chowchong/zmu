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
        Merge two SRT files based on track order.
        track_order: ['original', 'translated'] or ['translated', 'original']
        """
        import pysrt
        
        subs_orig = pysrt.open(original_path)
        subs_trans = pysrt.open(translated_path)
        
        # Ensure lengths match roughly (using original as base)
        merged_subs = pysrt.SubRipFile()
        
        limit = min(len(subs_orig), len(subs_trans))
        
        for i in range(limit):
            orig = subs_orig[i]
            trans = subs_trans[i]
            
            text_orig = orig.text
            text_trans = trans.text
            
            # Combine based on order
            parts = []
            for track in track_order:
                if track == 'original':
                    parts.append(text_orig)
                elif track == 'translated':
                    parts.append(text_trans)
            
            combined_text = "\n".join(parts)
            
            item = pysrt.SubRipItem(
                index=orig.index,
                start=orig.start,
                end=orig.end,
                text=combined_text
            )
            merged_subs.append(item)
            
        merged_subs.save(output_path, encoding='utf-8')
        print(f"✨ 合并字幕已保存: {output_path}")
        return output_path