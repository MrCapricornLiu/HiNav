#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# 读取原始文件
with open('/data/lch_zjc/VLN/ExtractObject/objects_predictedbygrok.json', 'r') as f:
    objects_data = json.load(f)

# 读取scan和instruction数据
with open('/data/lch_zjc/VLN/ExtractObject/extracted_scan_instruction.json', 'r') as f:
    scan_instruction_data = json.load(f)

# 转换为键值对格式
result = []
for i, item in enumerate(objects_data):
    # 获取对应的scan和instruction信息
    # 确保i不超出scan_instruction_data的范围
    scan_info = scan_instruction_data[i] if i < len(scan_instruction_data) else {"scan": "", "instruction": ""}
    
    keypair = {
        'scan': scan_info["scan"],
        'instruction': scan_info["instruction"],
        'direct_objects': item[0],
        'predicted_objects': item[1]
    }
    result.append(keypair)

# 写入新文件
with open('/data/lch_zjc/VLN/ExtractObject/objects_keypairs.json', 'w') as f:
    json.dump(result, f, indent=2)

print('转换完成，已生成包含scan和instruction的键值对格式文件') 