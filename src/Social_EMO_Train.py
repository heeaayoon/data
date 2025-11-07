# Social_EMO_Train.py
# 1단계와 2단계 모델을 모두 학습시킨 후 models/ 폴더에 저장

import os
import pandas as pd
import numpy as np
import torch
import pytz
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm

from datasets import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Social_EMO_Utils.py에서 공통 클래스 및 변수 가져오기
from Social_EMO_Utils import CustomModelWithAttention, MODEL_NAME, MAX_LEN

# 기본 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', MODEL_NAME)

# 경로 설정
BASE_PATH = Path("./") # 현재 폴더 기준
DATA_PATH = BASE_PATH / "data"
CSV_FILE_PATH = DATA_PATH / "Labeled_Social_Data_Augmented_4Class.csv"

# 출력 폴더 설정 (한국 시간 기준)
KST = pytz.timezone('Asia/Seoul')
RUN_ID = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
BASE_OUT_DIR = BASE_PATH / f"run_results_{RUN_ID}"
MODEL_SAVE_DIR = BASE_PATH / "models"
OUTPUT_DIR_STAGE1 = BASE_OUT_DIR / "stage1_training_output"
OUTPUT_DIR_STAGE2 = BASE_OUT_DIR / "stage2_training_output"
BEST1_DIR = MODEL_SAVE_DIR / "stage1_best_model"
BEST2_DIR = MODEL_SAVE_DIR / "stage2_best_model"

# 데이터 로더 클래스
class DataCollatorForChunkedText:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list) -> dict:
        all_chunks, all_labels, article_ids = [], [], []
        for i, feature in enumerate(features):
            all_chunks.extend(feature['content'])
            article_ids.extend([i] * len(feature['content']))
            all_labels.append(feature['labels'])

        inputs = self.tokenizer(
            all_chunks, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt"
        )
        inputs['labels'] = torch.tensor(all_labels, dtype=torch.long)
        inputs['article_ids'] = torch.tensor(article_ids, dtype=torch.long)
        return inputs

# 슬라이딩 윈도우(청크) 함수
def chunk_content(df):
    processed_data = []
    CHUNK_SIZE = MAX_LEN
    OVERLAP = 128
    STRIDE = CHUNK_SIZE - OVERLAP

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Chunking..."):
        tokens = tokenizer.encode(row['content'], add_special_tokens=False)
        content_list = [row['content']]
        if len(tokens) > CHUNK_SIZE:
            content_list = [tokenizer.decode(tokens[i:i+CHUNK_SIZE]) for i in range(0, len(tokens), STRIDE)]
        processed_data.append({'content': content_list, 'labels': row['labels']})
    return pd.DataFrame(processed_data)

# 성능 평가 함수
def compute_metrics(eval_pred: tuple) -> dict:
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print("\n--- Validation Report ---")
    print(classification_report(labels, preds, zero_division=0))
    print("-" * 25)
    return {"accuracy": acc, "macro_f1": f1}

# Trainer 생성 함수
def make_trainer(df: pd.DataFrame, num_labels: int, output_dir: str, num_epochs: int = 5) -> Trainer:
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=42)
    model = CustomModelWithAttention(MODEL_NAME, num_labels).to(device)

    labels_np = df["labels"].to_numpy(dtype="int64")
    class_counts = np.bincount(labels_np, minlength=num_labels)
    class_counts[class_counts == 0] = 1
    class_weights = torch.tensor(len(labels_np) / (num_labels * class_counts), dtype=torch.float)
    model.class_weights = class_weights
    print(f"\nApplied Class Weights: {class_weights}\n")

    args = TrainingArguments(
        output_dir=output_dir, save_strategy="epoch", eval_strategy="epoch",
        logging_strategy="steps", logging_steps=300, per_device_train_batch_size=4,
        num_train_epochs=num_epochs, learning_rate=1.5e-5, weight_decay=0.05,
        warmup_ratio=0.1, max_grad_norm=1.0, load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, fp16=torch.cuda.is_available(), report_to="none",
        remove_unused_columns=False, seed=42,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)]

    return Trainer(
        model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["test"],
        data_collator=DataCollatorForChunkedText(tokenizer),
        compute_metrics=compute_metrics, callbacks=callbacks
    )

def main():
    print(f"Using device: {device}")
    
    # 폴더 생성
    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    BEST1_DIR.mkdir(exist_ok=True)
    BEST2_DIR.mkdir(exist_ok=True)
    print(f"\n학습 결과는 {BASE_OUT_DIR} 에, 최종 모델은 {MODEL_SAVE_DIR} 에 저장됩니다.\n")

    # 데이터 로드 및 전처리
    df_original = pd.read_csv(CSV_FILE_PATH).reset_index(drop=True)
    df_original['content'] = df_original['content'].replace(r'^\\s*$', np.nan, regex=True)
    df_original = df_original.dropna(subset=['content', 'new_label']).reset_index(drop=True)
    df_original = df_original.drop_duplicates(subset=['content'], keep='first').reset_index(drop=True)
    VALID4 = {'중립','찬성_개정강화','반대_현상유지','찬성_폐지완화'}
    df_original = df_original[df_original['new_label'].isin(VALID4)].reset_index(drop=True)
    
    # 1단계 모델 데이터셋 준비
    print("\n--- Preparing Data for Stage 1 Model (3-Class) ---")
    label_map_s1 = {'찬성_개정강화': 0, '찬성_폐지완화': 1, '반대_현상유지': 1, '중립': 2}
    df_stage1 = df_original.copy()
    df_stage1['labels'] = df_stage1['new_label'].map(label_map_s1)
    df_stage1_train = chunk_content(df_stage1[['content', 'labels']])

    # 2단계 모델 데이터셋 준비
    print("\n--- Preparing Data for Stage 2 Model (2-Class) ---")
    label_map_s2 = {'찬성_폐지완화': 0, '반대_현상유지': 1}
    df_stage2 = df_original[df_original['new_label'].isin(label_map_s2.keys())].copy()
    df_stage2['labels'] = df_stage2['new_label'].map(label_map_s2)
    df_stage2_train = chunk_content(df_stage2[['content', 'labels']])

    # 1단계 모델 학습
    print("\n--- Training Stage 1 Model (3-Class) ---")
    trainer1 = make_trainer(df_stage1_train, num_labels=3, output_dir=str(OUTPUT_DIR_STAGE1), num_epochs=5)
    trainer1.train()
    trainer1.save_model(str(BEST1_DIR))
    print(f"\n--- Stage 1 Best Model Saved at {BEST1_DIR} ---")
    
    # 2단계 모델 학습
    print("\n--- Training Stage 2 Model (2-Class) ---")
    trainer2 = make_trainer(df_stage2_train, num_labels=2, output_dir=str(OUTPUT_DIR_STAGE2), num_epochs=5)
    trainer2.train()
    trainer2.save_model(str(BEST2_DIR))
    print(f"\n--- Stage 2 Best Model Saved at {BEST2_DIR} ---")
    
    print("\nAll training finished!")

if __name__ == "__main__":
    main()