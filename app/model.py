import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import boto3
import os

MODEL_DIR = "./fine_tuned_model"

# class TinyBERTService:
#     def __init__(self, cfg):
#         self.device = torch.device(cfg.device)
#         self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             cfg.model_path
#         ).to(self.device)
#         self.model.eval()

#         self.labels = ["negative", "positive"]

#     def predict(self, texts):
#         enc = self.tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=128,
#             return_tensors="pt"
#         )
#         enc = {k: v.to(self.device) for k, v in enc.items()}

#         with torch.no_grad():
#             logits = self.model(**enc).logits
#             probs = torch.softmax(logits, dim=-1)

#         return probs.cpu().tolist()

    # def map_labels(self, probs):
    #     results = []
    #     for row in probs:
    #         results.append({
    #             label: float(prob)
    #             for label, prob in zip(self.labels, row)
    #         })
    #     return results


def download_model_from_s3(bucket_name: str, prefix: str):
    s3 = boto3.client("s3")
    os.makedirs(MODEL_DIR, exist_ok=True)

    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    for obj in objects.get("Contents", []):
        key = obj["Key"]
        filename = key.split("/")[-1]
        if filename:
            s3.download_file(bucket_name, key, os.path.join(MODEL_DIR, filename))

class TinyBERTService:
    def __init__(self, bucket_name: str, prefix: str):
        if not os.path.exists(MODEL_DIR):
            download_model_from_s3(bucket_name, prefix)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.eval()
        self.labels = ["negative", "positive"]

    def predict(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)

        return probs.cpu().tolist()

    def map_labels(self, probs):
        results = []
        for row in probs:
            results.append({
                label: float(prob)
                for label, prob in zip(self.labels, row)
            })
        return results