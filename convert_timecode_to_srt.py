#!/usr/bin/env python3
"""
SRT格式转换工具
将非标准时间码格式转换为标准SRT格式

使用方法:
  python convert_timecode_to_srt.py input.srt output.srt
"""

import sys
import re
from pathlib import Path


def convert_timecode_to_srt(input_path, output_path):
    """
    转换时间码格式
    从: 00:00:00:00 - 00:00:01:17 (帧格式)
    到: 00:00:00,000 --> 00:00:01,170 (SRT标准格式)
    
    假设帧率为25fps (PAL标准)，可根据需要调整
    """
    FPS = 25  # 帧率，可根据实际视频调整（PAL=25, NTSC=30）
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    output_lines = []
    subtitle_index = 1
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 检测时间码行格式: HH:MM:SS:FF - HH:MM:SS:FF
        timecode_pattern = r'(\d{2}):(\d{2}):(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2}):(\d{2}):(\d{2})'
        match = re.match(timecode_pattern, line)
        
        if match:
            # 提取时间组件
            h1, m1, s1, f1 = match.groups()[:4]
            h2, m2, s2, f2 = match.groups()[4:]
            
            # 将帧转换为毫秒
            ms1 = int(f1) * 1000 // FPS
            ms2 = int(f2) * 1000 // FPS
            
            # 构建标准SRT时间码
            start_time = f"{h1}:{m1}:{s1},{ms1:03d}"
            end_time = f"{h2}:{m2}:{s2},{ms2:03d}"
            
            # 写入序号
            output_lines.append(f"{subtitle_index}\n")
            
            # 写入时间码
            output_lines.append(f"{start_time} --> {end_time}\n")
            
            # 收集字幕文本（直到空行）
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i])
                i += 1
            
            # 写入文本
            output_lines.extend(text_lines)
            
            # 添加空行分隔
            output_lines.append("\n")
            
            subtitle_index += 1
        
        i += 1
    
    # 保存转换后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"✅ 转换完成!")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_path}")
    print(f"   共 {subtitle_index - 1} 条字幕")


def main():
    if len(sys.argv) != 3:
        print("使用方法: python convert_timecode_to_srt.py <input.srt> <output.srt>")
        print("\n示例: python convert_timecode_to_srt.py video_CN.srt video_CN_converted.srt")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not Path(input_path).exists():
        print(f"❌ 错误: 文件不存在: {input_path}")
        sys.exit(1)
    
    convert_timecode_to_srt(input_path, output_path)


if __name__ == "__main__":
    main()
