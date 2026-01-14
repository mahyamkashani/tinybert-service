from fastapi import FastAPI
from app.schemas import InferenceRequest, LabelMapResponse
from app.config import InferenceConfig
from app.model import TinyBERTService

app = FastAPI()

cfg = InferenceConfig(max_length=128)
service = TinyBERTService(bucket_name="tinybert-model-bucket", prefix="fine_tuned_model/")

@app.get("/")
def root():
    return {"message": "TinyBERT API is running"}

@app.post("/predict", response_model=LabelMapResponse)
def predict(req: InferenceRequest):
    probs = service.predict(req.texts)
    mapped = service.map_labels(probs)
    return LabelMapResponse(predictions=mapped)
