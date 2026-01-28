#!/usr/bin/env python3
"""
Subgen CLI - 命令行入口
使用方法:
  python main.py install           # 安装模型
  python main.py check             # 检查状态
  python main.py transcribe <file> # 生成字幕
"""

import sys
import argparse
from pathlib import Path
from model_manager import ModelManager
from subtitle_engine import SubtitleEngine

def main():
    parser = argparse.ArgumentParser(description="Subgen - 智能字幕生成器")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # Install command
    subparsers.add_parser('install', help='安装必需模型')
    
    # Check command
    subparsers.add_parser('check', help='检查模型状态')
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser('transcribe', help='为视频生成字幕')
    transcribe_parser.add_argument('file', help='视频文件路径')
    transcribe_parser.add_argument('--model', '-m', help='指定模型 (funasr, whisper_small, whisper_medium, whisper_large)')
    transcribe_parser.add_argument('--lang', '-l', help='指定语言 (zh, en)')
    transcribe_parser.add_argument('--output', '-o', help='输出 SRT 路径')

    args = parser.parse_args()
    
    if args.command == 'install':
        print("📥 正在安装必需模型...")
        manager = ModelManager()
        manager.install_required_models()
        
    elif args.command == 'check':
        print("🔍 检查系统状态...")
        manager = ModelManager()
        status = manager.check_installation()
        for k, v in status.items():
            print(f"{'✅' if v else '❌'} {k}")
            
    elif args.command == 'transcribe':
        video_path = args.file
        if not Path(video_path).exists():
            print(f"❌ 文件不存在: {video_path}")
            sys.exit(1)
            
        output_path = args.output
        if not output_path:
            output_path = str(Path(video_path).with_suffix('.srt'))
            
        try:
            engine = SubtitleEngine()
            result = engine.transcribe(
                video_path, 
                model_key=args.model,
                language=args.lang
            )
            
            engine.save_srt(result, output_path, video_path=video_path)
            print(f"✨ 字幕已保存: {output_path}")
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
