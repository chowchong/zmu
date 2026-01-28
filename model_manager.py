"""
Subgen - 智能模型管理系统
支持自动下载、版本检查、离线使用
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, List
import shutil


class ModelManager:
    """模型管理器"""
    
    # 模型配置
    MODELS = {
        'funasr': {
            'name': 'FunASR Paraformer (实验性)',
            'model_id': 'iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            'required': False,  # Changed from True
            'size_mb': 220,
            'languages': ['zh', 'zh-CN'],
            'best_for': '⚠️ 实验性功能 - VAD过于严格导致识别不完整',
            'provider': 'modelscope'
        },
        'sensevoice': {
            'name': 'SenseVoice Small (实验性)',
            'model_id': 'iic/SenseVoiceSmall',
            'required': False,
            'size_mb': 80,
            'languages': ['zh', 'en', 'ja', 'ko', 'yue'],
            'best_for': '⚠️ 实验性功能 - 不支持时间戳，无法生成字幕',
            'provider': 'modelscope'
        },
        'whisper_small': {
            'name': 'Whisper Small (多语言)',
            'model_id': 'small',
            'required': True,
            'size_mb': 244,
            'languages': ['en', 'multi'],
            'best_for': '英文内容、多语言混合',
            'provider': 'openai'
        },
        'whisper_medium': {
            'name': 'Whisper Medium (高质量)',
            'model_id': 'medium',
            'required': True,
            'size_mb': 769,
            'languages': ['en', 'multi'],
            'best_for': '高质量英文识别',
            'provider': 'openai'
        },
        'whisper_large': {
            'name': 'Whisper Large-v3 (最强)',
            'model_id': 'large-v3',
            'required': False,
            'size_mb': 1550,
            'languages': ['all'],
            'best_for': '复杂口音、专业术语',
            'provider': 'openai'
        }
    }
    
    def __init__(self):
        """初始化模型管理器"""
        import sys
        
        # Check if running in frozen mode (App Bundle)
        if getattr(sys, 'frozen', False):
            # Running as compiled app
            # sys.executable is inside MacOS/ likely, so we go up to Contents/Resources
            # Example: App.app/Contents/MacOS/App -> App.app/Contents/Resources
            base_path = Path(sys.executable).parent.parent / 'Resources'
            self.models_dir = base_path / 'models'
            self.cache_dir = base_path / 'cache' # Local cache inside app
        else:
            # Running from source
            self.models_dir = Path.home() / '.subgen' / 'models'
            self.cache_dir = Path.home() / '.cache'
            
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # We might need to handle cache dir creation carefully if in read-only bundle, 
        # but user wants to manage it, so we assume write access or user action.
        
        self.config_file = self.models_dir / 'config.json'
        # Ensure config file exists or copy from resources if strictly needed, 
        # but for now we follow existing logic.
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """加载配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'installed_models': {}}
    
    def _save_config(self):
        """保存配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def check_installation(self) -> Dict[str, bool]:
        """检查所有模型安装状态"""
        status = {}
        for key, info in self.MODELS.items():
            if info['provider'] == 'openai':
                status[key] = self._check_whisper_model(info['model_id'])
            elif info['provider'] == 'modelscope':
                status[key] = self._check_funasr_model(info['model_id'])
        return status
    
    def _check_whisper_model(self, model_id: str) -> bool:
        """检查 Whisper 模型"""
        try:
            model_path = self.cache_dir / 'whisper' / f"{model_id}.pt"
            return model_path.exists()
        except:
            return False
    
    def _check_funasr_model(self, model_id: str) -> bool:
        """检查 FunASR 模型"""
        try:
            # Check default path (modelscope/model_id)
            base_cache = self.cache_dir / 'modelscope'
            path1 = base_cache / model_id
            if path1.exists():
                return True
                
            # Check hub path (modelscope/hub/model_id_with_underscore) - legacy/alternative
            path2 = base_cache / 'hub' / model_id.replace('/', '_')
            if path2.exists():
                return True
                
            return False
        except:
            return False

    def _download_whisper_model(self, model_id: str):
        """下载 Whisper 模型"""
        print(f"    ⏳ 下载 Whisper {model_id} 模型...")
        try:
            import whisper
            # Redirect download to our cache dir
            # Whisper download_root defaults to ~/.cache/whisper
            download_root = str(self.cache_dir / 'whisper')
            Path(download_root).mkdir(parents=True, exist_ok=True)
            
            # We can use whisper._download directly or set env var?
            # Easiest is to rely on load_model's download_root param if available,
            # or pre-download using torch.hub.download_url_to_file logic manually?
            # Actually whisper.load_model accepts 'download_root'.
            
            model = whisper.load_model(model_id, download_root=download_root)
            del model  # 释放内存
            print(f"    📥 模型已缓存")
        except Exception as e:
            raise Exception(f"Whisper 模型下载失败: {e}")
    
    def _download_funasr_model(self, model_id: str):
        """下载 FunASR 模型"""
        print(f"    ⏳ 下载 FunASR 模型...")
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            model_dir = snapshot_download(
                model_id,
                cache_dir=str(self.cache_dir / 'modelscope')
            )
            print(f"    📥 模型已缓存")
        except Exception as e:
            raise Exception(f"FunASR 模型下载失败: {e}")
    
    def get_recommended_model(self, language: Optional[str] = None) -> str:
        """智能推荐模型"""
        if language and language.startswith('zh'):
            return 'funasr'
        return 'whisper_medium'
    
    def list_installed_models(self) -> List[Dict]:
        """列出已安装的模型"""
        status = self.check_installation()
        installed = []
        for key, is_installed in status.items():
            if is_installed:
                info = self.MODELS[key].copy()
                info['key'] = key
                installed.append(info)
        return installed
    
    def install_required_models(self):
        """安装所有必需模型"""
        print("\n" + "="*60)
        print("🚀 Subgen 首次运行：正在安装必需模型")
        print("="*60 + "\n")
        
        status = self.check_installation()
        required_models = [k for k, v in self.MODELS.items() if v['required']]
        to_install = [k for k in required_models if not status.get(k, False)]
        
        if not to_install:
            print("✅ 所有必需模型已安装\n")
            return
        
        total_size = sum(self.MODELS[k]['size_mb'] for k in to_install)
        print(f"📦 需要下载 {len(to_install)} 个模型，总大小约 {total_size} MB\n")
        
        for idx, model_key in enumerate(to_install, 1):
            model_info = self.MODELS[model_key]
            print(f"[{idx}/{len(to_install)}] {model_info['name']}")
            print(f"    大小: {model_info['size_mb']} MB")
            print(f"    用途: {model_info['best_for']}\n")
            
            try:
                self.download_model(model_key)
                print(f"    ✅ 安装成功\n")
            except Exception as e:
                print(f"    ❌ 安装失败: {e}\n")
                raise
        
        print("="*60)
        print("🎉 所有必需模型安装完成！")
        print("="*60 + "\n")
    
    def download_model(self, model_key: str):
        """下载指定模型"""
        if model_key not in self.MODELS:
            raise ValueError(f"未知模型: {model_key}")
        
        model_info = self.MODELS[model_key]
        
        if model_info['provider'] == 'openai':
            self._download_whisper_model(model_info['model_id'])
        elif model_info['provider'] == 'modelscope':
            self._download_funasr_model(model_info['model_id'])
        
        self.config['installed_models'][model_key] = {'version': '1.0'}
        self._save_config()
    
    
    def _download_funasr_model(self, model_id: str):
        """下载 FunASR 模型"""
        print(f"    ⏳ 下载 FunASR 模型...")
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            model_dir = snapshot_download(
                model_id,
                cache_dir=str(Path.home() / '.cache' / 'modelscope')
            )
            print(f"    📥 模型已缓存")
        except Exception as e:
            raise Exception(f"FunASR 模型下载失败: {e}")
    
    def get_recommended_model(self, language: Optional[str] = None) -> str:
        """智能推荐模型"""
        if language and language.startswith('zh'):
            return 'funasr'
        return 'whisper_medium'
    
    def list_installed_models(self) -> List[Dict]:
        """列出已安装的模型"""
        status = self.check_installation()
        installed = []
        for key, is_installed in status.items():
            if is_installed:
                info = self.MODELS[key].copy()
                info['key'] = key
                installed.append(info)
        return installed


def main():
    """命令行工具"""
    import sys
    
    manager = ModelManager()
    
    if len(sys.argv) < 2:
        print("📦 Subgen - 模型管理工具\n")
        print("用法:")
        print("  python model_manager.py check    # 检查模型状态")
        print("  python model_manager.py install  # 安装必需模型")
        return
    
    command = sys.argv[1]
    
    if command == 'check':
        print("🔍 检查模型安装状态...\n")
        status = manager.check_installation()
        for key, info in manager.MODELS.items():
            icon = "✅" if status.get(key, False) else "❌"
            print(f"  {icon} {info['name']} ({key})")
    
    elif command == 'install':
        manager.install_required_models()
    
    else:
        print(f"未知命令: {command}")

if __name__ == "__main__":
    main()
    def _download_model_with_progress(self, model_key: str, progress_callback=None):
        """下载模型并报告进度（用于后台下载）"""
        if model_key not in self.MODELS:
            return False
        
        model_info = self.MODELS[model_key]
        model_type = model_info['type']
        
        try:
            if model_type == 'whisper':
                # For Whisper, we'll use the existing download method
                # Report progress at milestones since whisper doesn't provide granular progress
                if progress_callback:
                    progress_callback(0, model_info['size_mb'] * 1024 * 1024)
                
                self._download_whisper_model(model_info['id'])
                
                if progress_callback:
                    progress_callback(model_info['size_mb'] * 1024 * 1024, model_info['size_mb'] * 1024 * 1024)
                
                return True
                
            elif model_type == 'funasr':
                if progress_callback:
                    progress_callback(0, model_info['size_mb'] * 1024 * 1024)
                
                self._download_funasr_model(model_info['id'])
                
                if progress_callback:
                    progress_callback(model_info['size_mb'] * 1024 * 1024, model_info['size_mb'] * 1024 * 1024)
                
                return True
            
            return False
            
        except Exception as e:
            print(f"下载失败: {e}")
            return False
