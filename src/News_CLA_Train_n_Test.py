# News_CLA_Train_n_Test.py
# train/test
import os
import pickle

import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.functional import softmax

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModel,
    TrainingArguments, Trainer,
    EarlyStoppingCallback
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import shutil

FILE_PATH_WITH_FEATURES = "../data/processed/5000_w_label.csv"
MODEL_NAME = "klue/roberta-base"
MAX_LENGTH = 512
SLIDING_STRIDE = 256
OTHER_NAME = "기타"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로더(인코딩 견고)
def load_data_with_robust_encodings(file_path):
    encodings = ['utf-8', 'euc-kr', 'cp949']
    last_err = None
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise Exception(f"파일 로드 실패: {file_path} / 마지막 오류: {last_err}")

# 데이터 로드
df = load_data_with_robust_encodings(FILE_PATH_WITH_FEATURES).copy()

# content/라벨 필수 컬럼 체크
required_cols = {"content", "category", "MAIN_NAME"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"필수 컬럼 누락: {missing}")

df["content"] = df["content"].astype(str)
df["category"] = df["category"].astype(str)
df["MAIN_NAME"] = df["MAIN_NAME"].astype(str)
df["__index"] = df.index  # 문서 식별자

print(df["category"].unique())
print(df["MAIN_NAME"].unique())

tmp_main = df["MAIN_NAME"].astype(str)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=tmp_main
)

# 학습에서만 '기타' 제외
train_df = train_df[train_df["MAIN_NAME"] != OTHER_NAME]

# 라벨 맵 (기타 제외 기준)
category_list = sorted(train_df["category"].astype(str).unique().tolist())
mainname_list = sorted(train_df["MAIN_NAME"].astype(str).unique().tolist())

law2id = {l: i for i, l in enumerate(category_list)}
cat2id = {c: i for i, c in enumerate(mainname_list)}
id2law = {v: k for k, v in law2id.items()}
id2cat = {v: k for k, v in cat2id.items()}

# 학습 라벨 부여
train_df["labels_law"] = train_df["category"].map(law2id)
train_df["labels_cat"] = train_df["MAIN_NAME"].map(cat2id)

# 평가셋: 공정 비교 위해 '기타' 샘플 제외 권장
test_df_eval = test_df[test_df["MAIN_NAME"] != OTHER_NAME].copy()
test_df_eval["labels_law"] = test_df_eval["category"].map(law2id)
test_df_eval["labels_cat"] = test_df_eval["MAIN_NAME"].map(cat2id)

# 혹시 train엔 없고 test에만 있는 클래스가 있으면 NaN → 제거
before = len(test_df_eval)
test_df_eval = test_df_eval.dropna(subset=["labels_law", "labels_cat"])
test_df_eval["labels_law"] = test_df_eval["labels_law"].astype(int)
test_df_eval["labels_cat"] = test_df_eval["labels_cat"].astype(int)
if len(test_df_eval) < before:
    print(f"[주의] 평가셋에서 train에 없는 클래스 {before - len(test_df_eval)}개 제거됨")

num_laws = len(law2id)
num_cats = len(cat2id)

# 법안-중분류 허용 마스크(기타 제외 기준)
allowed_cats_by_law = {
    law2id[l]: set(
        train_df[train_df["category"] == l]["labels_cat"].unique().tolist()
    )
    for l in category_list
}
mask_mat = torch.full((num_laws, num_cats), fill_value=-1e9, dtype=torch.float)
for law_id, cats in allowed_cats_by_law.items():
    for c in cats:
        mask_mat[law_id, c] = 0.0


# 토크나이저 & 슬라이딩 토큰화
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_with_sliding(batch):
    tokenized_batch = tokenizer(
        batch["content"],
        truncation=True, padding=False, max_length=MAX_LENGTH,
        stride=SLIDING_STRIDE, return_overflowing_tokens=True, return_offsets_mapping=False,
    )
    labels_law_list, labels_cat_list, doc_index_list = [], [], []
    # 입력 배치의 각 샘플이 여러 청크로 늘어날 수 있으므로 매핑 사용
    for sample_index in tokenized_batch["overflow_to_sample_mapping"]:
        labels_law_list.append(batch["labels_law"][sample_index])
        labels_cat_list.append(batch["labels_cat"][sample_index])
        doc_index_list.append(batch["__index"][sample_index])

    tokenized_batch["labels_law"] = labels_law_list
    tokenized_batch["labels_cat"] = labels_cat_list
    tokenized_batch["doc_index"]  = doc_index_list
    del tokenized_batch["overflow_to_sample_mapping"]
    return tokenized_batch

# Dataset 생성 및 map 
train_dataset = Dataset.from_pandas(
    train_df[["content", "labels_law", "labels_cat", "__index"]].copy()
)
test_dataset = Dataset.from_pandas(
    test_df_eval[["content", "labels_law", "labels_cat", "__index"]].copy()
)

cols_remove_train = train_dataset.column_names
cols_remove_test  = test_dataset.column_names

train_dataset = train_dataset.map(
    tokenize_with_sliding, batched=True,
    remove_columns=cols_remove_train, batch_size=256
)
test_dataset = test_dataset.map(
    tokenize_with_sliding, batched=True,
    remove_columns=cols_remove_test, batch_size=64
)


# 클래스 가중치(기타 제외 학습 분포 기준)
class_weights_cat_np = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["labels_cat"]),
    y=train_df["labels_cat"]
)
class_weights_cat = torch.tensor(class_weights_cat_np, dtype=torch.float)

# 모델 정의
class HierarchicalClassifier(nn.Module):
    def __init__(self, model_name, num_laws, num_cats, mask_mat, class_weights_cat=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

       
        self.config = self.encoder.config

        self.dropout = nn.Dropout(p=0.3)
        self.law_head = nn.Linear(hidden, num_laws)
        self.cat_head = nn.Linear(hidden, num_cats)

        self.register_buffer("mask_mat", mask_mat)
        if class_weights_cat is not None:
            self.register_buffer("class_weights_cat", class_weights_cat)
        else:
            self.class_weights_cat = None

    
    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.encoder.set_input_embeddings(new_embeddings)

    def forward(self, input_ids, attention_mask, labels_law=None, labels_cat=None, use_predicted_law_for_mask=False):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = enc.last_hidden_state[:, 0]
        features = self.dropout(pooled)

        law_logits = self.law_head(features)
        cat_logits_raw = self.cat_head(features)

        if labels_law is not None and not use_predicted_law_for_mask:
            law_for_mask = labels_law
        else:
            law_for_mask = law_logits.argmax(dim=-1)

        
        mask_selected = self.mask_mat[law_for_mask]  # index_select 대신 직접 인덱싱        
        cat_logits = cat_logits_raw + mask_selected

        loss = None
        if (labels_law is not None) and (labels_cat is not None):
            loss_law = nn.CrossEntropyLoss()(law_logits, labels_law)
            if self.class_weights_cat is not None:
                loss_cat = nn.CrossEntropyLoss(weight=self.class_weights_cat)(cat_logits, labels_cat)
            else:
                loss_cat = nn.CrossEntropyLoss()(cat_logits, labels_cat)
            loss = loss_law + loss_cat 

        return {"loss": loss, "law_logits": law_logits, "cat_logits": cat_logits, "cat_logits_raw": cat_logits_raw}

    
def custom_collator_dynamic(batch):
    text_inputs = [{k: v for k, v in item.items() if k in ["input_ids", "attention_mask"]} for item in batch]
    tokenized_batch = tokenizer.pad(text_inputs, padding=True, return_tensors="pt", max_length=MAX_LENGTH)
    return {
        **tokenized_batch,
        "labels_law": torch.tensor([b["labels_law"] for b in batch], dtype=torch.long),
        "labels_cat": torch.tensor([b["labels_cat"] for b in batch], dtype=torch.long),
    }

# 커스텀 Trainer (WeightedRandomSampler)
class HierTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        labels_cat = np.array(self.train_dataset["labels_cat"])

        unique_labels, counts = np.unique(labels_cat, return_counts=True)
        class_weights_map = {label: 1.0 / count for label, count in zip(unique_labels, counts)}
        sample_weights = np.array([class_weights_map[label] for label in labels_cat])
        sample_weights = torch.from_numpy(sample_weights).double()

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_law = inputs.pop("labels_law")
        labels_cat = inputs.pop("labels_cat")
        outputs = model(**inputs, labels_law=labels_law, labels_cat=labels_cat, use_predicted_law_for_mask=False)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels_law = inputs.pop("labels_law")
        labels_cat = inputs.pop("labels_cat")
        with torch.no_grad():
            outputs = model(**inputs, labels_law=labels_law, labels_cat=labels_cat, use_predicted_law_for_mask=False)
        law_logits = outputs["law_logits"].detach().cpu()
        cat_logits = outputs["cat_logits"].detach().cpu()
        logits = torch.cat([law_logits, cat_logits], dim=1)
        labels = torch.stack([labels_law.cpu(), labels_cat.cpu()], dim=1)
        loss = outputs["loss"].detach().cpu()
        return (loss, logits, labels)


# 문서단위 메트릭 (청크 -> 문서 평균 후 Argmax)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    law_logits = logits[:, :num_laws]
    cat_logits = logits[:, num_laws:]

    law_probs = softmax(torch.tensor(law_logits), dim=-1).numpy()
    cat_probs = softmax(torch.tensor(cat_logits), dim=-1).numpy()

   
    doc_indices = test_dataset["doc_index"]

    doc_law_probs, doc_cat_probs = defaultdict(list), defaultdict(list)
    for pL, pC, idx in zip(law_probs, cat_probs, doc_indices):
        doc_law_probs[idx].append(pL)
        doc_cat_probs[idx].append(pC)

    final_law_preds, final_cat_preds = [], []
    final_law_labels, final_cat_labels = [], []

    for idx in doc_law_probs.keys():
        avg_law = np.mean(doc_law_probs[idx], axis=0)
        avg_cat = np.mean(doc_cat_probs[idx], axis=0)
        final_law_preds.append(int(np.argmax(avg_law)))
        final_cat_preds.append(int(np.argmax(avg_cat)))

        
        row = test_df_eval.loc[test_df_eval["__index"] == idx]
        if len(row) == 0:
            
            continue
        final_law_labels.append(int(row["labels_law"].iloc[0]))
        final_cat_labels.append(int(row["labels_cat"].iloc[0]))

    acc_law = accuracy_score(final_law_labels, final_law_preds)
    f1_law = f1_score(final_law_labels, final_law_preds, average="macro")
    acc_cat = accuracy_score(final_cat_labels, final_cat_preds)
    f1_cat = f1_score(final_cat_labels, final_cat_preds, average="macro")
    f1_cat_weighted = f1_score(final_cat_labels, final_cat_preds, average="weighted")

    return {
        "accuracy_law": acc_law,
        "f1_law": f1_law,
        "accuracy_cat": acc_cat,
        "f1_cat": f1_cat,
        "f1_cat_weighted": f1_cat_weighted
    }


# 모델/학습 세팅
model = HierarchicalClassifier(
    MODEL_NAME, num_laws=num_laws, num_cats=num_cats,
    mask_mat=mask_mat, class_weights_cat=class_weights_cat
).to(device)

training_args = TrainingArguments(
    output_dir="./results_hier_weighted",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1.5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),   
    logging_dir="./logs_hier_weighted",
    logging_steps=50,
    metric_for_best_model="f1_cat",
    greater_is_better=True,
    label_smoothing_factor=0.1,  
    dataloader_pin_memory=True,  # GPU 사용 시 메모리 고정
    dataloader_drop_last=True,   # 마지막 불완전한 배치 제거         
)

trainer = HierTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=custom_collator_dynamic,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 학습 시작
train_output = trainer.train()
print("\n[완료] 학습 종료. best checkpoint가 로드되었습니다.")

# best checkpoint를 ./results_hier_weighted/best_model 폴더에 복사
best_ckpt = trainer.state.best_model_checkpoint
best_model_dir = "./results_hier_weighted/best_model"

if best_ckpt is not None:
    shutil.copytree(best_ckpt, best_model_dir, dirs_exist_ok=True)
    print(f"[저장 완료] Best model을 {best_model_dir} 에 복사했습니다.")
else:
    print("[경고] best_model_checkpoint가 없습니다. TrainingArguments에 load_best_model_at_end=True 가 설정되었는지 확인하세요.")

# 검증 실행
eval_output = trainer.evaluate()
print("[검증 결과]", eval_output)

print("\n[알림] 학습은 '기타' 제외, 평가는 '기타' 제외 샘플로 점수 산정했습니다.")
print("[팁] 추후 추론 서비스에서 '기타'를 쓰고 싶다면, 확률 임계값 기반 reject-option을 추가하면 됩니다.")
save_dir = "./results_hier_weighted"
os.makedirs(save_dir, exist_ok=True)

label_info = {
    "law2id": law2id,
    "id2law": id2law,
    "cat2id": cat2id,
    "id2cat": id2cat,
    "mask_mat": mask_mat.cpu().numpy(),
    "category_list": category_list,
    "mainname_list": mainname_list,
}

with open(os.path.join(save_dir, "label_mapping.pkl"), "wb") as f:
    pickle.dump(label_info, f)