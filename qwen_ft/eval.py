import datasets
import numpy as np
import torch
from torch import nn

from qwen_ft.metric import metrics_compute


def to_batch(x, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]


def unpack(x):
    assert isinstance(x, torch.Tensor), type(x)
    return x.detach().float().cpu().numpy().tolist()

import re


def extract_speaker_info(text: str) -> str:
    """
    Extracts the speaker ID and indices from the audio tags in the given text,
    and returns a string formatted as "speakerID_index1_index2".
    """
    # Find all audio tag contents
    audio_paths = re.findall(r'<audio>(.*?)</audio>', text)
    
    # Extract base filenames without extension
    filenames = [path.split('/')[-1].rsplit('.', 1)[0] for path in audio_paths]
    
    # Split into speaker ID and index, then collect them
    speaker_id = filenames[0].split('_')[0]
    indices = [name.split('_')[1] for name in filenames]
    
    return f"{speaker_id}_{indices[0]}_{indices[1]}"


def eval_model(
    model: nn.Module,
    ds: datasets.Dataset,
    eval_batch_size: int = 16,
    target_label: str = "severe_osa",
    num_classes: int = 2,
) -> None:
    """
    This function evaluates the accuracy of a given model on a given dataset.

    Parameters:
    model (nn.Module): The model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label, predicted label,
                    accuracy of prediction, logits and soft label for each example in the dataset.
    metric (dict): A metric dict including accuracy, f1, precision, recall.
    """

    model.eval()

    with torch.no_grad():
        results = []
        # all_preds = []
        # all_labels = []
        # for ex in ds:
        for batch in to_batch(ds, eval_batch_size):
            # print(torch.tensor(batch["input_ids"]).squeeze(0).shape)
            # print(torch.tensor(batch["input_features"]).squeeze(0).shape)
            # print(torch.tensor(batch["attention_mask"]).squeeze(0).shape)
            # print(torch.tensor(batch["feature_attention_mask"]).squeeze(0).shape)
            # exit()

            labels = batch[target_label]
            
            raw_logits, raw_scores, _ = model(
                input_ids=torch.tensor(batch["input_ids"]).squeeze(0),
                input_features=torch.tensor(batch["input_features"]).squeeze(0),
                attention_mask=torch.tensor(batch["attention_mask"]).squeeze(0),
                feature_attention_mask=torch.tensor(batch["feature_attention_mask"]).squeeze(0),
            )
            probs = torch.nn.functional.softmax(raw_logits, dim=-1)
            logits = unpack(raw_logits)

            preds = np.argmax(probs.cpu(), axis=-1)
            # labels = np.argmax(labels, axis=-1)
            results.extend(
                [
                    dict(
                        # txt=batch['text'],
                        # wav_path=wav_path,
                        # id=wav_path.split('/')[-1].split('.')[0],
                        id=extract_speaker_info(query),
                        # input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        # acc=label == pred,
                        logits=logit,
                        soft_label=prob,
                    )
                    for query, label, pred, prob, logit in zip(
                        batch["query"], labels, preds, probs, logits
                    )
                ]
            )
            break
        # accs = [r["acc"] for r in results]
        all_preds = [r["hard_label"] for r in results]
        all_labels = [r["gt_label"] for r in results]
        print(all_labels)
        print(all_preds)
        # print("Accuracy:", np.mean(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))
        metric = metrics_compute(all_preds, all_labels, num_classes=num_classes)
        print("Test metric:", metric)

    model.train()
    return datasets.Dataset.from_list(results), metric
