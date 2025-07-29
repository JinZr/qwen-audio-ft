# import debugpy

# # 5678是debugpy服务器监听的端口号，确保这个端口在你的系统上是空闲的
# debugpy.listen(('0.0.0.0', 5679))
# print("⏳ Waiting for debugger to attach...")

# # 让debugpy等待VSCode的调试器连接
# debugpy.wait_for_client()
# print("🚀 Debugger attached!")


import argparse
import itertools
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_optimizer as toptim
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence

import qwen_ft.logger as logger
from qwen_ft.common import get_tokenizer
from qwen_ft.dataset import tokenize_dataset
from qwen_ft.eval import eval_model
from qwen_ft.loss import FocalLoss
from qwen_ft.metric import metrics_compute
from qwen_ft.model import TransformerWithHead
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    get_scheduler,
    set_seed,
)
from transformers.modeling_utils import load_sharded_checkpoint

custom_kwargs = {
    "trust_remote_code": True,
    "bf16": torch.cuda.is_bf16_supported(),
    "fp32": not torch.cuda.is_bf16_supported(),
    # "torch_dtype": 'auto',
}


def main(
    args,
    model_path: str = "Qwen-Audio",
    train_json: str = "/mnt/nvme_share/cuizy/SA-detection/scp/whisper/02/sr_level/train_data_augment.json",
    dev_json: str = "/mnt/nvme_share/cuizy/SA-detection/scp/whisper/02/val_data.json",
    test_json: str = "/mnt/nvme_share/cuizy/SA-detection/scp/whisper/02/test_data.json",
    target_label: str = "severe_osa",
    num_class: int = 4,
    save_path: str = "./exp/",
    force_retrain: bool = False,
    mode: str = "lora",
    seed: int = 224,
    max_ctx: int = 512,
    batch_size: int = 8,
    mini_batch_size: int = 4,
    learning_rate: float = 1e-4,
    optim: str = "AdamW",
    lr_schedule: str = "linear",
    weight_decay: float = 0.005,
    num_train_epochs: int = 2,
    warmup_step_frac: float = 0.2,
    evaluation_steps: int = 100,
    logging_steps: int = 20,
    model_parallel: bool = True,
):
    set_seed(seed)
    logger.configure(
        name="{target_label}_{model_name}_{mode}_{learning_rate}_{lr_schedule}_{batch_size}-{mini_batch_size}_{seed}",
        target_label=target_label,
        model_path=model_path,
        mode=mode,
        learning_rate=learning_rate,
        save_path=save_path,
        seed=seed,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        lr_schedule=lr_schedule,
        epochs=num_train_epochs,
    )
    dataset = load_dataset(
        "json", data_files={"train": train_json, "dev": dev_json, "test": test_json}
    )
    train_ds, dev_ds, test_ds = dataset["train"], dataset["dev"], dataset["test"]
    # train_ds = train_ds.map(lambda ex: {"soft_label": [1 - float(ex["label"]), float(ex["label"])]})
    train_ds = train_ds.shuffle(seed=seed)
    # dev_ds = dev_ds.map(lambda ex: {"soft_label": [1 - float(ex["label"]), float(ex["label"])]})
    # test_ds = test_ds.map(lambda ex: {"soft_label": [1 - float(ex["label"]), float(ex["label"])]})
    tokenizer = get_tokenizer(model_path)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    train_ds = tokenize_dataset(
        train_ds, tokenizer=tokenizer, processor=processor, max_ctx=max_ctx
    )
    dev_ds = tokenize_dataset(
        dev_ds, tokenizer=tokenizer, processor=processor, max_ctx=max_ctx
    )
    test_ds = tokenize_dataset(
        test_ds, tokenizer=tokenizer, processor=processor, max_ctx=max_ctx
    )

    def maybe_load_model(model):
        if os.path.exists(os.path.join(save_path, "results.pkl")) and not force_retrain:
            # print("loading from", save_path)
            # checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
            # if not os.path.exists(checkpoint_path):
            #     # Assume this means we have a sharded checkpoint, and load it appropriately
            #     load_sharded_checkpoint(model, checkpoint_path)
            # else:
            #     state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
            #     state_dict = {
            #         k.replace("transformer.module", "transformer"): v
            #         for (k, v) in state_dict.items()
            #     }
            #     custom_kwargs["state_dict"] = state_dict
            checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
            adapter_path = os.path.join(save_path, "adapter_model.safetensors")
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path))
                print("Best checkpoint loaded from pytorch_model")
            elif os.path.exists(adapter_path):
                model = model.get_base_model()
                model = PeftModel.from_pretrained(model, save_path)
                print("Best checkpoint loaded from adapter")
            elif not os.path.exists(adapter_path):
                load_sharded_checkpoint(model, save_path)
                print("Best checkpoint loaded from shards")
            return True
        return False

    already_trained = False
    if model_parallel:
        model = TransformerWithHead.from_pretrained(
            model_path=model_path,
            num_class=num_class,
            device_map="auto",
            # local_files_only=True,
            # **custom_kwargs,
        )
                # Ensure model parameters are cast to bfloat16 if supported
        # if custom_kwargs.get("bf16", False):
        #     model = model.to(torch.bfloat16)
        if mode == "lora":
            peft_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
                modules_to_save=["score", "diagnosis"],
                lora_dropout=args.lora_dropout,
                use_dora=args.use_dora,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        already_trained = maybe_load_model(model)
    else:
        # TODO: data parallel
        pass

    if already_trained:
        test_results, test_metric = eval_model(
            model,
            test_ds,
            eval_batch_size=1,
            target_label=target_label,
            num_classes=num_class,
        )
    else:
        start_time = time.time()
        nsteps = len(train_ds) * num_train_epochs // batch_size

        if args.loss == "xent":
            loss_fn = torch.nn.CrossEntropyLoss()
        elif args.loss == "focal":
            loss_fn = FocalLoss()

        if optim.lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optim.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optim.lower() == "adafactor":
            optimizer = toptim.Adafactor(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        lr_scheduler = get_scheduler(
            name=lr_schedule,
            optimizer=optimizer,
            num_warmup_steps=int(warmup_step_frac * nsteps),
            num_training_steps=nsteps,
        )

        step = 0
        it = itertools.chain.from_iterable(itertools.repeat(train_ds, num_train_epochs))
        losses = []
        pred_logging = []
        label_logging = []
        best_metric = 0
        io_device = model.device if hasattr(model, "device") else 0
        scaler = GradScaler()

        while step < nsteps:
            loss_batch = 0
            if evaluation_steps and step % evaluation_steps == 0:
                eval_results, eval_metric = eval_model(
                    model,
                    dev_ds,
                    eval_batch_size=1,
                    target_label=target_label,
                    num_classes=num_class,
                )
                logger.logkvs(eval_metric)
                if best_metric == 0 or best_metric <= eval_metric["f1"]:
                    best_metric = eval_metric["f1"]
                    (
                        model if hasattr(model, "save_pretrained") else model.module
                    ).save_pretrained(
                        # save_path, safe_serialization=False
                        save_path
                    )
                    print("Model saved of step", step)

            all_logits = []
            all_labels = []
            for i in range(batch_size // mini_batch_size):
                try:
                    mbatch = [next(it) for _ in range(mini_batch_size)]
                except StopIteration:
                    break

                labels = torch.tensor([ex[target_label] for ex in mbatch]).to(io_device)
                
                # Pad text sequences
                ids = [torch.tensor(ex["input_ids"], dtype=torch.long).transpose(0, 1) for ex in mbatch]
                masks = [torch.tensor(ex["attention_mask"], dtype=torch.long).transpose(0, 1) for ex in mbatch]
                padded_ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(io_device)
                input_ids = padded_ids.transpose(1, 2).squeeze(1)  # (batch, feature_dim, seq_len)
                padded_attention_mask = pad_sequence(masks, batch_first=True, padding_value=0).to(io_device)
                attention_mask = padded_attention_mask.transpose(1, 2).squeeze(1)

                # Pad audio features
                feats = [torch.tensor(ex["input_features"], dtype=torch.float).transpose(0, 1) for ex in mbatch]
                fmask = [torch.tensor(ex["feature_attention_mask"], dtype=torch.long).transpose(0, 1) for ex in mbatch]
                
                padded_feats = pad_sequence(feats, batch_first=True, padding_value=0.0).to(io_device)
                input_features = padded_feats.transpose(1, 2)  # (batch, 2, feature_dim, seq_len)
                # print(input_features.shape)
                input_features = input_features.flatten(0, 1)
                # print(input_features.shape)

                padded_feature_attention_mask = pad_sequence(fmask, batch_first=True, padding_value=0).to(io_device)
                feature_attention_mask = padded_feature_attention_mask.transpose(1, 2).squeeze(1)

                # ---- Truncate to the minimal required sequence length ----
                # Use the attention mask to find the longest real (non‑padded) sequence
                with torch.no_grad():
                    max_len = feature_attention_mask.sum(dim=-1).max().item()
                # Slice both the features and the mask so downstream layers
                # do not process unnecessary padding
                input_features = input_features[..., :max_len]
                feature_attention_mask = feature_attention_mask[..., :max_len].squeeze(0)
                # print(input_features.shape)
                # print(feature_attention_mask.shape)

                # print(input_ids.shape)
                # print(input_features.shape)
                # print(attention_mask.shape)
                # print(feature_attention_mask.shape)
                with autocast():  
                    logits, scores, _ = model(
                        input_ids=input_ids,
                        input_features=input_features,
                        attention_mask=attention_mask,
                        feature_attention_mask=feature_attention_mask,
                    )
                    print(logits)
                    exit()
                all_logits.extend(logits.to(io_device))
                pred_logging.extend(torch.argmax(F.softmax(logits, dim=-1), dim=-1))
                all_labels.extend(labels)
                label_logging.extend(labels)

            print(all_labels)
            print(all_logits)
            all_logits = torch.stack(all_logits)
            all_labels = torch.stack(all_labels)
            print(all_labels)
            print(all_logits)
            exit()
            loss = loss_fn(all_logits, all_labels)
            loss_batch += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss_batch)
            logger.logkvs(
                {
                    "step": step,
                    "progress": step / nsteps,
                    "loss": loss_batch,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            )
            # logger.logkvs(metrics_compute(torch.argmax(F.softmax(all_logits, dim=-1), dim=-1).cpu(), all_labels.cpu(), num_classes=num_class))
            # metrics.append(metrics_compute(all_logits, all_labels, num_classes=num_class))
            optimizer.zero_grad()
            lr_scheduler.step()
            if logging_steps and step % logging_steps == 0:
                pred_logging = torch.stack(pred_logging).cpu()
                label_logging = torch.stack(label_logging).cpu()
                metric = metrics_compute(pred_logging, label_logging, num_class)
                print(
                    f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} {metric} {len(losses)}"
                )
                losses = []
                pred_logging = []
                label_logging = []
            step += 1
            logger.dumpkvs()

        checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
        adapter_path = os.path.join(save_path, "adapter_model.safetensors")
        # if os.path.exists(adapter_path):
        #     model.load_state_dict(torch.load(adapter_path))
        #     print("Best checkpoint loaded from adapter")
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print("Best checkpoint loaded from pytorch_model")
        elif os.path.exists(adapter_path):
            model = model.get_base_model()
            model = PeftModel.from_pretrained(model, save_path)
            print("Best checkpoint loaded from adapter")
        elif not os.path.exists(adapter_path):
            load_sharded_checkpoint(model, save_path)
            print("Best checkpoint loaded from shards")

        final_eval_results = None
        print("Final evaluation:")
        test_results, test_metric = eval_model(
            model,
            test_ds,
            eval_batch_size=1,
            target_label=target_label,
            num_classes=num_class,
        )
        logger.logkvs(test_metric)
        logger.dumpkvs()
        print("Model training took", time.time() - start_time, "seconds")

    with open(os.path.join(save_path, "results.pkl"), "wb") as f:
        pickle.dump(
            {
                "test_metric": test_metric,
                "test_results": test_results,
            },
            f,
        )
    logger.shutdown()

    with open(
        os.path.join(
            save_path,
            "results_summary.json",
        ),
        "a+",
    ) as f:
        json.dump(test_metric, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen-Audio")
    parser.add_argument("--model_path", type=str, default="Qwen-Audio")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_ctx", type=int, default=256)
    parser.add_argument("--mode", type=str, default="lora", choices=["default", "lora"])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--train_json",
        type=str,
        default="/home/jinzr/hdd/projects/qwen-audio-ft/data/fold1_train.json",
    )
    parser.add_argument(
        "--dev_json",
        type=str,
        default="/home/jinzr/hdd/projects/qwen-audio-ft/data/fold1_val.json",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="/home/jinzr/hdd/projects/qwen-audio-ft/data/fold1_val.json",
    )
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--use_dora", action="store_true")
    parser.add_argument("--seed", type=int, default=224)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument(
        "--save_path",
        type=str,
        default="./exp/qwen2.5omni",
    )
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--target_label", type=str, default="sr_level")
    parser.add_argument("--num_class", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--warmup_step_frac", type=float, default=0.2)
    parser.add_argument("--evaluation_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--loss", type=str, default="xent", choices=["xent", "focal"])
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # fire.Fire(main)
    args = parse_args()
    # if Path(args.save_path).exists():
        # raise FileExistsError(f"{args.save_path} already exists!")
    # else:
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    print(custom_kwargs)
    with open(os.path.join(args.save_path, "model_config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    os.system(
        "cp {} {}".format(
            "qwen-audio-ft/train.py", os.path.join(args.save_path, "train.py")
        )
    )

    main(
        args=args,
        model_path=args.model_path,
        train_json=args.train_json,
        dev_json=args.dev_json,
        test_json=args.test_json,
        target_label=args.target_label,
        num_class=args.num_class,
        save_path=args.save_path,
        force_retrain=args.force_retrain,
        mode=args.mode,
        seed=args.seed,
        max_ctx=args.max_ctx,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        optim=args.optim,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_step_frac=args.warmup_step_frac,
        evaluation_steps=args.evaluation_steps,
        logging_steps=args.logging_steps,
    )
