from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import nn

from transformers import (
    AutoConfig,
    PreTrainedModel,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)


@dataclass
class HeadOutput:
    logits: torch.FloatTensor


class TransformerWithHead(PreTrainedModel):
    def __init__(self, model_path, num_class, **kwargs):
        config = AutoConfig.from_pretrained(model_path, **kwargs)
        super().__init__(config)
        self.num_class = num_class
        lm = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **kwargs)
        self.transformer = lm.thinker
        # print(self.transformer)
        # A processor to convert raw text + audio into model inputs
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        hidden_size = getattr(
            getattr(getattr(config, "thinker_config"), "text_config"), "hidden_size"
        )
        self.diagnosis = nn.Linear(hidden_size, self.num_class)
        self.score = nn.Linear(hidden_size, 1)
        # torch.nn.init.normal_(self.score.weight, std=0.0)
        # torch.nn.init.normal_(self.diagnosis.weight, std=0.0)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        return cls(model_path, **kwargs)

    def forward(
        self,
        input_ids, input_features, attention_mask, feature_attention_mask,
        **processor_kwargs,
    ):
        """
        Forward pass that directly accepts raw text prompts and their paired audio
        waveforms, leveraging ``Qwen2_5OmniProcessor`` to build the inputs expected
        by the Qwen2.5‑Omni backbone.

        Parameters
        ----------
        texts : Union[str, List[str]]
            A single prompt or a batch of prompts.
        audios : Union[np.ndarray, torch.Tensor, List]
            The corresponding audio waveforms for each prompt (1‑D float arrays in
            the range [-1, 1]).
        sampling_rate : int, default 16000
            Sampling rate of the provided audio waveforms.
        processor_kwargs : dict
            Extra keyword arguments forwarded to ``self.processor``.

        Returns
        -------
        logits : torch.FloatTensor
            Classification logits of shape (batch_size, num_class).
        hidden_states : torch.FloatTensor
            Pooled hidden states before the classification head.
        """

        # Move tensors to the same device as the backbone
        # input_ids = input_ids.to(self.transformer.device)

        # Forward through the backbone
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]  # [batch_size, seq_len, hidden_size]

        # input_lens = (attention_mask != 0).sum(dim=-1)

        hidden_states = transformer_outputs.mean(1)

        # Classification head
        logits = self.diagnosis(hidden_states)
        scores = self.score(hidden_states)

        return logits, scores, hidden_states



if __name__ == "__main__":
    model = TransformerWithHead(
        model_path="/home/jinzr/nfs/projects/qwen-audio-ft/Qwen2.5-Omni-7B", num_class=2
    )
