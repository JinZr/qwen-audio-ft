#!/usr/bin/env python3
"""
extract_ahi.py

把 JSONL 记录中的 AHI 指数和“重度 OSA”标志拆分成独立字段。

用法:
    python extract_ahi.py input.jsonl output.jsonl
"""

import argparse
import json
import re
import sys
from typing import List

# AHI 后面常出现的格式: “AHI 是 45.9” / “AHI: 45.9” 等
AHI_PATTERN   = re.compile(
    r'AHI\s*(?:值|指数)?\s*(?:[:：]?|(?:是|为))\s*([0-9]+(?:\.[0-9]+)?)'
)
SEVERE_PATTERN = re.compile(r'非重度\s*OSA', re.IGNORECASE)

# 从 query 字段中提取 <audio>xxx.wav</audio> 路径
AUDIO_PATTERN = re.compile(r'<audio>(.*?)</audio>')

def enrich(rec: dict) -> dict:
    """在原记录上新增 AHI 与 severe_osa 字段"""
    blob = ' '.join(str(rec.get(k, '')) for k in ('response', 'query', 'system'))
    m = AHI_PATTERN.search(blob)
    if not m:
        raise ValueError(f'未找到 AHI: {blob[:120]}')
    rec = rec.copy()
    rec['AHI'] = float(m.group(1))
    rec['severe_osa'] = 0 if SEVERE_PATTERN.search(blob) else 1
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input',  help='原始 JSONL')
    ap.add_argument('output', help='输出 JSONL')
    ap.add_argument('--train', help='若指定，则额外生成符合 train_data.json 格式的文件')

    args = ap.parse_args()

    with open(args.input,  'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        train_records = []  # 收集生成 train_data.json 所需的条目（按 query 字段）
        for line in fin:
            rec = json.loads(line)
            try:
                rec = enrich(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                if args.train:
                    # 直接使用原始 query 字段构造 train_data.json 项
                    train_records.append({
                        'query': rec.get('query', ''),
                        'response': rec.get('response', ''),
                        'label': rec['severe_osa'],
                        'severe_osa': rec['severe_osa'],
                        'AHI': rec['AHI'],
                        'start': 0,
                        'end': 0,
                        'duration': 0,
                        'language_id': 0,
                        'task_id': 0
                    })
            except ValueError as err:
                print('警告:', err, file=sys.stderr)
                continue  # 必要时可改为 raise
        # 如有需要，输出 train_data.json
        if args.train:
            with open(args.train, 'w', encoding='utf-8') as tf:
                json.dump(train_records, tf, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()