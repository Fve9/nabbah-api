from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import torch
import json
import os

app = FastAPI()

MODEL_ID = "fve9/Nabbah_saudi_bert"
HF_TOKEN = os.getenv("HF_TOKEN")  # مهم إذا الريبو Private

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
model.eval()

mapping_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="label_mapping.json",
    repo_type="model",
    token=HF_TOKEN
)

with open(mapping_path, "r", encoding="utf-8") as f:
    mapping = json.load(f)

id2label = {int(k): v for k, v in mapping["id2label"].items()}

class ComplaintInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "API is working"}

@app.post("/predict")
def predict(data: ComplaintInput):
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        score = probs[0][pred_id].item()

    return {
        "label": id2label[pred_id],
        "score": round(score, 4)
    }