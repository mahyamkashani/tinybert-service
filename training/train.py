import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from pathlib import Path

model_path = Path(__file__).resolve().parent.parent / "training" / "fine_tuned_model"

MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
NUM_LABELS = 2
OUTPUT_DIR = "./fine_tuned_model"

def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    dataset = load_dataset("imdb")  # binary classification
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(500))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        num_train_epochs=2,
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
