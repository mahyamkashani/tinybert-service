from pydantic import BaseModel, Field
from typing import List, Dict

class InferenceRequest(BaseModel):
    texts: List[str] = Field(min_length=1)

class LabelMapResponse(BaseModel):
    predictions: List[Dict[str, float]]
