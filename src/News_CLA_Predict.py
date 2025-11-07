# News_CLA_Predict.py
# Predict - Hierarchical 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pickle

MODEL_NAME = "klue/roberta-base"
BEST_MODEL_PATH = "../models/news_model/model.safetensors"
INPUT_CSV = "../data/processed/news_pre.csv"   # content 컬럼 포함한 데이터
OUTPUT_CSV = "../data/processed/news_predict_result.csv"
LABEL_PATH = "../models/news_model/label_mapping.pkl"

MAX_LENGTH = 256
STRIDE = 128
BATCH_SIZE = 8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(LABEL_PATH, "rb") as f:
    label_info = pickle.load(f)

law2id = label_info["law2id"]
id2law = label_info["id2law"]
cat2id = label_info["cat2id"]
id2cat = label_info["id2cat"]
mask_mat = torch.tensor(label_info["mask_mat"])
category = label_info["category_list"]
MAIN_NAME = label_info["mainname_list"]

num_laws = len(category)
num_cats = len(MAIN_NAME)

# 모델 정의
class HierarchicalClassifier(nn.Module):
    def __init__(self, model_name, num_laws, num_cats):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.law_head = nn.Linear(hidden, num_laws)
        self.cat_head = nn.Linear(hidden, num_cats)

    def forward(self, input_ids, attention_mask):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = enc.last_hidden_state[:, 0]
        law_logits = self.law_head(pooled)
        cat_logits = self.cat_head(pooled)   # mask 적용 X
        return {"law_logits": law_logits, "cat_logits": cat_logits, "embedding": pooled}

# 모델 로드
model = HierarchicalClassifier(MODEL_NAME, num_laws, num_cats).to(device)
state_dict = load_file(BEST_MODEL_PATH, device=str(device))
model.load_state_dict(state_dict, strict=False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 데이터 로드
df = pd.read_csv(INPUT_CSV)
texts = df["content"].astype(str).tolist()

# 예측
final_law_preds, final_cat_preds = [], []

for text in tqdm(texts, desc="Hierarchical 추론"):
    tokens = tokenizer(text, truncation=False, padding=False)
    input_ids_full = tokens["input_ids"]

    # 긴 문서 → 여러 chunk
    chunks = []
    for start in range(0, len(input_ids_full), MAX_LENGTH-STRIDE):
        end = min(start+MAX_LENGTH, len(input_ids_full))
        chunks.append(input_ids_full[start:end])
        if end == len(input_ids_full): break

    # 문서 단위 logits 모으기
    all_law_logits, all_cat_logits = [], []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i+BATCH_SIZE]
        enc = tokenizer.pad({"input_ids": batch_chunks}, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids, attention_mask)
            all_law_logits.append(out["law_logits"].cpu())
            all_cat_logits.append(out["cat_logits"].cpu())

    # 문서 단위 평균 확률
    law_avg = torch.mean(torch.cat(all_law_logits, dim=0), dim=0)
    law_id = torch.argmax(law_avg).item()   # 최종 대분류 확정

    cat_avg = torch.mean(torch.cat(all_cat_logits, dim=0), dim=0)
    allowed_cats = (mask_mat[law_id] == 0).nonzero(as_tuple=True)[0]

    if len(allowed_cats) > 0:
        rel_idx = torch.argmax(cat_avg[allowed_cats]).item()
        cat_id = allowed_cats[rel_idx].item()
    else:
        cat_id = -1 

    final_law_preds.append(law_id)
    final_cat_preds.append(cat_id)

# 결과 저장
df["predicted_category"] = [id2law[i] for i in final_law_preds] # 대분류 예측결과
df["predicted_MAIN_NAME"] = [id2cat[i] for i in final_cat_preds] # 중분류 예측결과
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"저장 완료: {OUTPUT_CSV}")