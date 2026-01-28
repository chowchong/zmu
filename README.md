# zmu
Subtitle generator using Whisper &amp; FunASR model

# Subgen 使用指南

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 安装模型
```bash
# 推荐：只安装 Whisper（最可靠）
python main.py install
# 系统会自动安装必需的 Whisper 模型
```

### 3. 生成字幕
```bash
# 推荐方式（默认使用 Whisper Medium）
python main.py transcribe video.mp4

# 或使用 GUI
python gui.py
```

## 模型选择

### ✅ 推荐模型

**Whisper Medium** (默认)
- ✅ **完整识别**: 100% 识别所有语音
- ✅ **精确时间轴**: 适合字幕生成
- ✅ **多语言支持**: 中英日韩等 99 种语言
- ⚠️ **方言识别**: 对新马口音/方言可能有误识别

**使用建议**: 
- 主力使用 Whisper 生成字幕
- 编辑 `~/.subgen/user_dict.txt` 添加方言词汇修正
- 使用内置编辑器手动调整

### ⚠️ 实验性模型（不推荐）

**FunASR Paraformer**
- ❌ **识别不完整**: VAD 过于严格，只识别约 10% 内容
- ✅ **方言优势**: 对新马口音识别准确
- 💡 **适用场景**: 仅用于方言测试对比

**SenseVoice Small**
- ❌ **无时间戳**: 无法生成带时间轴的字幕
- ✅ **情感识别**: 支持情感标注
- 💡 **适用场景**: 仅用于纯文本转录

## 常见问题

### Q: 方言识别不准确怎么办？
A: 编辑用户词典
```bash
# 编辑 ~/.subgen/user_dict.txt
咖啡店: kopitiam
巴刹: pasar
```

### Q: 想测试不同模型效果？
A: 使用 --model 参数（会自动生成编号版本）
```bash
python main.py transcribe video.mp4 --model whisper_medium  # video.srt
python main.py transcribe video.mp4 --model funasr          # video_1.srt
```

### Q: GUI 在哪里选择模型？
A: 目前通过命令行 --model 参数，GUI 模型选择功能正在开发中。
