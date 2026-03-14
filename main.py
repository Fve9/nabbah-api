import os
import json
import re
from typing import List, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# ======================================
# 1) MODEL
# ======================================
MODEL_NAME = "Fve9/Nabbah_saudi_bert"
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer_file = hf_hub_download(
    repo_id=MODEL_NAME,
    filename="tokenizer.json",
    repo_type="model",
    token=HF_TOKEN
)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tokenizer_file,
    unk_token="[UNK]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    mask_token="[MASK]"
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN
)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

mapping_path = hf_hub_download(
    repo_id=MODEL_NAME,
    filename="label_mapping.json",
    repo_type="model",
    token=HF_TOKEN
)

with open(mapping_path, "r", encoding="utf-8") as f:
    mapping = json.load(f)

id2label = {int(k): v for k, v in mapping["id2label"].items()}

# ======================================
# 2) LIGHT STEMMING
# ======================================
LIGHT_STEM_REPLACEMENTS = [
    (r"\b(انقطع(ت|وا|نا)?|منقطع(ه)?|مقطوع(ه)?|مفصول(ه)?|طافي(ه)?|طفت|طفى|طفا)\b", "انقطاع"),
    (r"\b(تعطل(ت|وا|نا)?|معطل(ه)?|خربان(ه)?|خربت|وقف(ت|وا|نا)?|متوقف(ه)?|مايشتغل|ما يشتغل)\b", "توقف"),
    (r"\b(يحترق|احترق(ت|وا|نا)?|احتراق|حريق|ولع(ت|وا|نا)?|مولع)\b", "حريق"),
    (r"\b(تسرب|يسرب|سرب(ت|وا|نا)?|تهريب)\b", "تسرب"),
    (r"\b(هدد(ني|نا|هم)?|تهديد|يتوعد|توعد)\b", "تهديد"),
    (r"\b(تحرش|يتحرش|تحرش(وا|ت)?|اعتدى|اعتداء|اعتدا(ت|وا)?)\b", "اعتداء"),
    (r"\b(تسمم|مسمم|تسممت|سم(م)?|سموم)\b", "تسمم"),
]

def light_stem(text: str) -> str:
    text = str(text)
    for pat, rep in LIGHT_STEM_REPLACEMENTS:
        text = re.sub(pat, rep, text)
    return text

# ======================================
# 3) PRIORITY RULES
# ======================================
GLOBAL_URGENT_PATTERNS = [
    r"(?<!\w)(الان|الآن|الحين|فورا|بسرعة|ضروري|عاجل|طارئ|خطر|كارثة)(?!\w)",
    r"(حريق|انفجار|غرق)",
    r"(نزيف|فقدان وعي|اغماء|اختناق)",
    r"(تسمم)",
    r"(العنف|اعتداء|تحرش|تهديد|ابتزاز)",
    r"(محبوس|محتجز|عالق)",
    r"(حادث|اصابة|مصاب)",
    r"(انقطاع)",
    r"(تسرب)",
    r"(توقف)",
]

AUTHORITY_URGENT_PATTERNS = {
    "شركة الكهرباء السعودية": [
        r"(تماس كهربائي|سلك مكشوف|شرار|ماس|طافي)",
        r"(حريق)",
        r"(انقطاع)",
    ],
    "شركة الاتصالات السعودية": [
        r"(انقطاع)",
        r"(لا يوجد شبكة|مافي شبكة|بدون شبكة|ضعف شديد|طافي)",
        r"(توقف).*(انترنت|نت|خدمة)",
    ],
    "وزارة الصحة": [
        r"(حالة حرجة|حرجة|لا يوجد طبيب|لا يوجد اسعاف|رفض استقبال|خطأ طبي)",
        r"(نزيف|فقدان وعي|تسمم|اختناق)",
    ],
    "الجرائم المعلوماتية (كلنا أمن)": [
        r"(تحرش|ابتزاز|تهديد|اختراق|هكر|تسريب|نشر صور|احتيال|نصب)",
    ],
    "مكافحة مخدرات": [
        r"(مخدرات|ترويج|حشيش|شبو|كبتاجون)",
    ],
    "بلدي": [
        r"(سقوط مبنى|خطر انهيار|انهيار)",
        r"(حفرة خطيرة|صرف مكشوف|فيض|تجمع مياه)",
    ],
    "وزارة التعليم": [
        r"(اعتداء|تحرش|تنمر شديد)",
        r"(خطر على الطلاب|مدرسة غير آمنة|اصابة طالب)",
    ],
    "وزارة الحج والعمرة": [
        r"(مفقود|ضايع|تائه)",
        r"(تدافع|ازدحام شديد|خطر)",
        r"(حادث|اصابة|مصاب)",
    ],
    "وزارة الرياضة": [
        r"(تدافع|ازدحام شديد|خطر)",
        r"(اعتداء|تحرش)",
        r"(حادث|اصابة|مصاب)",
    ],
    "وزارة التجارة": [
        r"(فاسد|منتهي الصلاحية|غش|مغشوش|تلاعب غذائي)",
        r"(تسمم)",
    ],
    "وزارة البيئة": [
        r"(تلوث|نفوق|روائح كريهة|تسرب نفطي|مخلفات خطرة)",
        r"(حريق)",
    ],
    "الهيئة العامة للنقل": [
        r"(حادث|اصابة|مصاب|اصطدم|دهس)",
        r"(سائق متهور|طريق غير آمن)",
    ],
    "الأحوال المدنية": [
        r"(وفاة|بلاغ وفاة)",
        r"(هوية).*(مفقود|ضايعة|مسروقة)",
    ],
}

def classify_priority(text: str, authority: Optional[str]) -> str:
    text = str(text)

    for pat in GLOBAL_URGENT_PATTERNS:
        if re.search(pat, text):
            return "عالية"

    if authority:
        for pat in AUTHORITY_URGENT_PATTERNS.get(authority, []):
            if re.search(pat, text):
                return "عالية"

    return "منخفضة"

# ======================================
# 4) RESPONSE MODELS
# ======================================
class PredictedLabel(BaseModel):
    label: str
    confidence: float
    rank: int

class ComplaintRequest(BaseModel):
    text: str

class ComplaintResponse(BaseModel):
    text: str
    predicted_labels: List[PredictedLabel]
    current_label: Optional[str]
    priority: str
    status: str
    reassignment_count: int
    manual_review: bool

class ReassignRequest(BaseModel):
    text: str
    predicted_labels: List[PredictedLabel]
    current_label: Optional[str]
    priority: str
    status: str
    reassignment_count: int
    manual_review: bool

# ======================================
# 5) PREDICTION FUNCTION
# ======================================
def predict_complaint(text: str, top_k: int = 3):
    clean_text = light_stem(text)

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    num_labels = probs.shape[1]
    k = min(top_k, num_labels)
    top_probs, top_ids = torch.topk(probs, k=k, dim=1)

    predicted_labels = []
    for rank, (label_id, conf) in enumerate(
        zip(top_ids[0].tolist(), top_probs[0].tolist()),
        start=1
    ):
        predicted_labels.append(
            {
                "label": id2label[int(label_id)],
                "confidence": round(float(conf), 4),
                "rank": rank,
            }
        )

    current_label = predicted_labels[0]["label"] if predicted_labels else None
    priority = classify_priority(clean_text, current_label)

    return {
        "text": text,
        "predicted_labels": predicted_labels,
        "current_label": current_label,
        "priority": priority,
        "status": "assigned",
        "reassignment_count": 0,
        "manual_review": False,
    }

# ======================================
# 6) REASSIGN FUNCTION
# ======================================
def reassign_to_next(predicted_labels: List[dict], reassignment_count: int):
    next_index = reassignment_count + 1

    if next_index < len(predicted_labels):
        return {
            "current_label": predicted_labels[next_index]["label"],
            "status": "reassigned",
            "reassignment_count": reassignment_count + 1,
            "manual_review": False,
        }

    return {
        "current_label": None,
        "status": "manual_review",
        "reassignment_count": reassignment_count,
        "manual_review": True,
    }

# ======================================
# 7) FASTAPI APP
# ======================================
app = FastAPI(
    title="Nabbah Complaint Classifier API",
    description="API for classifying complaint authority and urgency",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ComplaintResponse)
def predict(request: ComplaintRequest):
    return predict_complaint(request.text)

@app.post("/reassign", response_model=ComplaintResponse)
def reassign(request: ReassignRequest):
    predicted_labels = [item.dict() for item in request.predicted_labels]
    result = reassign_to_next(predicted_labels, request.reassignment_count)

    return {
        "text": request.text,
        "predicted_labels": predicted_labels,
        "current_label": result["current_label"],
        "priority": request.priority,
        "status": result["status"],
        "reassignment_count": result["reassignment_count"],
        "manual_review": result["manual_review"],
    }