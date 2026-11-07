from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ntl.labels import LABELS

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SMOKE_STEPS = 8


@dataclass(frozen=True)
class LlmRun:
    base: str = "Qwen/Qwen3-4B-Base"
    max_length: int = 512
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 1e-4
    epochs: float = 1.0
    batch: int = 4
    accumulation: int = 4
    warmup: float = 0.03
    seed: int = 2027
    output: str = "/kaggle/working/treino"
    smoke: bool = False


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weight = inputs.pop("weight")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        losses = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, reduction="none")
        loss = (losses * weight).sum() / weight.sum()
        return (loss, outputs) if return_outputs else loss


def load(run: LlmRun):
    tokenizer = AutoTokenizer.from_pretrained(run.base)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        run.base, num_labels=len(LABELS), quantization_config=quant, device_map={"": 0}, dtype=torch.float16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        task_type="SEQ_CLS",
        r=run.rank,
        lora_alpha=run.alpha,
        lora_dropout=run.dropout,
        target_modules=TARGETS,
    )
    model = get_peft_model(model, lora)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model, tokenizer


def encoded(rows: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(rows.rename(columns={"label": "labels"}), preserve_index=False)
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def train(run: LlmRun, model, tokenizer, rows: pd.DataFrame):
    dataset = encoded(rows, tokenizer, run.max_length)
    steps = (
        SMOKE_STEPS if run.smoke else int(np.ceil(len(dataset) * run.epochs / (run.batch * run.accumulation)))
    )
    args = TrainingArguments(
        output_dir=run.output,
        seed=run.seed,
        per_device_train_batch_size=run.batch,
        gradient_accumulation_steps=run.accumulation,
        learning_rate=run.learning_rate,
        max_steps=steps,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(steps * run.warmup)),
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=max(1, steps // 50),
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=dataset, data_collator=DataCollatorWithPadding(tokenizer)
    )
    trainer.train()
    return trainer.model


def predict_proba(
    model, tokenizer, texts: Sequence[str], max_length: int = 512, batch: int = 16
) -> np.ndarray:
    model.eval()
    order = np.argsort([len(text) for text in texts])
    probabilities = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            chunk = order[start : start + batch]
            inputs = tokenizer(
                [texts[i] or " " for i in chunk],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            probabilities[chunk] = model(**inputs).logits.float().softmax(-1).cpu().numpy()
    return probabilities
