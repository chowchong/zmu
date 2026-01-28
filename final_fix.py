#!/usr/bin/env python3
"""最终修复脚本 - 使用预保存的正确代码"""

# 读取正确的方法实现
with open('/tmp/correct_methods.txt', 'r') as f:
    new_code = f.read()

# 读取现有文件
with open('gui.py.broken_backup', 'r') as f:
    lines = f.readlines()

# 找到 check_models 的位置
start_idx = None
for i, line in enumerate(lines):
    if '    def check_models(self):' in line:
        start_idx = i
        break

# 找到下一个方法
end_idx = None  
for i in range(start_idx + 1, len(lines)):
    if lines[i].startswith('    def ') and 'download' not in lines[i] and 'check_models' not in lines[i]:
        end_idx = i
        break

print(f"替换行 {start_idx + 1} 到 {end_idx}")

# 组装新文件
new_lines = lines[:start_idx] + [new_code + '\n'] + lines[end_idx:]

with open('gui.py', 'w') as f:
    f.writelines(new_lines)

print("✅ 完成")
