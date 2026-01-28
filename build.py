#!/usr/bin/env python3
"""
Subgen - macOS App Build Script
自动化打包流程，包括清理、构建、签名
"""

import subprocess
import shutil
import os
from pathlib import Path

def clean_build_artifacts():
    """清理之前的构建产物"""
    print("🧹 清理旧的构建文件...")
    
    folders_to_clean = ['build', 'dist']
    for folder in folders_to_clean:
        if Path(folder).exists():
            shutil.rmtree(folder)
            print(f"   已删除: {folder}/")
    
    print("✅ 清理完成\n")

def install_pyinstaller():
    """确保 PyInstaller 已安装"""
    print("📦 检查 PyInstaller...")
    try:
        import PyInstaller
        print(f"   已安装: PyInstaller {PyInstaller.__version__}\n")
    except ImportError:
        print("   未安装，正在安装...")
        subprocess.run(['pip', 'install', 'pyinstaller'], check=True)
        print("✅ PyInstaller 安装完成\n")

def build_app():
    """执行 PyInstaller 打包"""
    print("🚀 开始构建 Subgen.app...")
    print("=" * 60)
    
    cmd = ['pyinstaller', 'Subgen.spec', '--clean', '--noconfirm']
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print("\n❌ 构建失败")
        return False
    
    print("\n✅ 构建成功")
    return True

def create_resources_folders():
    """在 App Bundle 中创建 Resources 文件夹结构"""
    print("\n📁 创建 Resources 文件夹结构...")
    
    app_path = Path('dist/Subgen.app')
    resources_path = app_path / 'Contents' / 'Resources'
    
    # Create models and cache directories
    models_dir = resources_path / 'models'
    cache_dir = resources_path / 'cache'
    
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a README in models folder
    readme_content = """# Subgen Models Directory

此文件夹用于存储下载的 AI 模型。
程序运行时会自动下载所需的模型到这里。

您可以：
- 查看已下载的模型
- 手动删除不需要的模型以释放空间
- 手动添加预先下载的模型文件

模型文件结构：
- cache/whisper/        - Whisper 语音识别模型
- cache/modelscope/     - FunASR 模型
- cache/argos/          - Argos 翻译模型
"""
    
    (models_dir / 'README.txt').write_text(readme_content, encoding='utf-8')
    
    print(f"   ✅ 已创建: {models_dir}")
    print(f"   ✅ 已创建: {cache_dir}")

def sign_app():
    """对 App 进行临时签名（避免 macOS 安全警告）"""
    print("\n🔐 对 App 进行签名...")
    
    app_path = 'dist/Subgen.app'
    
    # Ad-hoc signing (for local use)
    cmd = ['codesign', '--force', '--deep', '--sign', '-', app_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ 签名成功")
    else:
        print(f"   ⚠️  签名失败（可选步骤）: {result.stderr}")

def main():
    """主构建流程"""
    print("\n" + "=" * 60)
    print("🎬 Subgen - macOS App 自动化构建")
    print("=" * 60 + "\n")
    
    # Step 1: Clean
    clean_build_artifacts()
    
    # Step 2: Check PyInstaller
    install_pyinstaller()
    
    # Step 3: Build
    if not build_app():
        return
    
    # Step 4: Create Resources structure
    create_resources_folders()
    
    # Step 5: Sign
    sign_app()
    
    # Final message
    print("\n" + "=" * 60)
    print("🎉 构建完成！")
    print("=" * 60)
    print(f"\nApp 位置: {Path('dist/Subgen.app').absolute()}")
    print("\n您可以:")
    print("  1. 双击运行 dist/Subgen.app")
    print("  2. 右键 -> Show Package Contents 查看内部结构")
    print("  3. 进入 Contents/Resources/models 管理模型文件")
    print("\n")

if __name__ == '__main__':
    # Change to script directory
    os.chdir(Path(__file__).parent)
    main()
