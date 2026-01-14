from pydantic import BaseModel, Field

class InferenceConfig(BaseModel):
    model_path: str = "./fine_tuned_model"
    max_length: int = Field(gt=0, le=512)
    device: str = "cpu"
