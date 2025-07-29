import functools
import os
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Optional

from datasets import Dataset
from pydub import AudioSegment
from qwen_omni_utils import process_mm_info

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


def tokenize_dataset(
    raw_ds: Dataset,
    processor: Qwen2_5OmniProcessor,
    tokenizer: Callable,
    max_ctx: int,
):
    def process_function(sample):
        # audio_url = sample["wav_path"]
        query = sample["query"]
        # === Build multimodal conversation: separate text and audio ===
        content_list = []
        remain = query
        while True:
            start = remain.find("<audio>")
            if start == -1:
                if remain:
                    content_list.append({"type": "text", "text": remain})
                break

            # Add text before the <audio> tag, if any
            if start > 0:
                content_list.append({"type": "text", "text": remain[:start]})

            end = remain.find("</audio>", start)
            if end == -1:
                # Malformed tag, treat the rest as plain text
                content_list.append({"type": "text", "text": remain[start:]})
                break

            # Extract the full audio path between tags
            audio_path = remain[start + len("<audio>") : end].strip()
            content_list.append({"type": "audio", "audio": audio_path})

            # Move past the closing </audio> tag
            remain = remain[end + len("</audio>") :]

        # If multiple audio segments are present, concatenate them with 2s silence
        # audio_items = [c for c in content_list if c["type"] == "audio"]
        # if len(audio_items) > 1:
        #     audio_paths = [item["audio"] for item in audio_items]
        #     # Build a unique filename based on input files
        #     base_names = [os.path.splitext(os.path.basename(p))[0] for p in audio_paths]
        #     concat_name = "_".join(base_names) + "_concat.wav"
        #     # Save in same directory as first audio
        #     concat_dir = os.path.dirname(audio_paths[0])
        #     os.makedirs(concat_dir, exist_ok=True)
        #     concat_path = os.path.join(concat_dir, concat_name)
        #     # Only concatenate if file doesn't already exist
        #     if not os.path.exists(concat_path):
        #         silence_seg = AudioSegment.silent(duration=2000)  # 2 seconds silence
        #         segments = []
        #         for p in audio_paths:
        #             segments.append(AudioSegment.from_file(p))
        #             segments.append(silence_seg)
        #         # Drop trailing silence
        #         segments = segments[:-1]
        #         combined = sum(segments)
        #         combined.export(concat_path, format="wav")
        #     # Replace all audio entries with a single concatenated audio entry
        #     content_list = [c for c in content_list if c["type"] != "audio"]
        #     content_list.append({"type": "audio", "audio": concat_path})

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "你是一名耳鼻喉科主治医师，专攻阻塞性睡眠呼吸暂停（OSA）。请通过对比同一患者坐位与仰卧位朗读同一段文本的语音差异，判断其是否患有重度 OSA。",
                    }
                ],
            },
            {
                "role": "user",
                "content": content_list,
            },
        ]

        # query = f"<audio>{audio_url}</audio>{prompt}"
        # audio_info = tokenizer.process_audio(query)
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )

        data = {k: v[0].tolist() for k, v in inputs.items()}
        data["query"] = query
        for key in ("severe_osa", "sr_level", "label"):
            if key in sample:
                data[key] = sample[key]

        return data

    ds = raw_ds.map(process_function, batched=False)
    return ds


# def tokenize_dataset(
#     raw_ds: Dataset,
#     tokenizer: Callable,
#     max_ctx: int,
# ):
#     """
#     This function prepares the dataset for training. It takes the raw dataset, a formatting function,
#     a tokenizer, a maximum context length

#     Parameters:
#     raw_ds: The raw dataset to be processed.
#     tokenizer: The tokenizer to be used on the formatted dataset.
#     max_ctx: The maximum context length for the tokenizer.

#     Returns:
#     ds: The processed and shuffled dataset ready for training.
#     """

#     def process_function(res):
#         toks = tokenizer(res["text"], max_length=max_ctx, truncation=True)
#         return dict(
#             input_ids=toks["input_ids"],
#         )

#     # ds = raw_ds.map(process_function, batched=False).filter(lambda x: len(x["input_ids"]) < max_ctx)
#     ds = raw_ds.map(process_function, batched=False)
#     return ds

if __name__ == "__main__":
    from common import get_tokenizer
    from datasets import load_dataset

    dataset = load_dataset(
        "json",
        data_files={
            "train": "/home/jinzr/nfs/projects/qwen-audio-ft/data/fold1_train.json",
        },
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(
        "/home/jinzr/nfs/projects/qwen-audio-ft/Qwen2.5-Omni-7B"
    )
    tokenizer = get_tokenizer("/home/jinzr/nfs/projects/qwen-audio-ft/Qwen2.5-Omni-7B")
    tokenize_dataset(dataset, tokenizer=tokenizer, processor=processor, max_ctx=2048)
