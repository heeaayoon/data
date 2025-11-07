# Social_EMO_Test.py
# 학습된 모델을 TestFile.csv로 평가하고, 정확도, F1-score 등의 종합 리포트를 생성

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import json
from pathlib import Path
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

# Social_EMO_Utils.py에서 공통 클래스 및 변수 가져오기
from Social_EMO_Utils import CustomModelWithAttention, MODEL_NAME, MAX_LEN

# 기본 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# 경로 설정
BASE_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_PATH / "models"
PROCESSED_PATH = BASE_PATH / "data" / "processed"
TEST_FILE_PATH = PROCESSED_PATH /"Social_EMO_TestFile.csv"
BEST1_MODEL_PATH = MODEL_PATH / "stage1_best_model" / "model.safetensors"
BEST2_MODEL_PATH = MODEL_PATH / "stage2_best_model" / "model.safetensors"

# 예측 함수 정의
def predict_two_stage(text, model1, model2, tokenizer, device):
    CHUNK_SIZE = MAX_LEN
    OVERLAP = 128
    STRIDE = CHUNK_SIZE - OVERLAP
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) < CHUNK_SIZE:
        content_list = [text]
    else:
        content_list = [tokenizer.decode(tokens[i:i+CHUNK_SIZE]) for i in range(0, len(tokens), STRIDE)]

    inputs = tokenizer(content_list, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    inputs['article_ids'] = torch.zeros(len(content_list), dtype=torch.long)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits1 = model1(**inputs)
        probs1 = F.softmax(logits1, dim=-1).squeeze()
        prob_positive, prob_negative, prob_neutral = probs1[0].item(), probs1[1].item(), probs1[2].item()

        logits2 = model2(**inputs)
        probs2  = F.softmax(logits2, dim=-1).squeeze()
        prob_repeal, prob_maintain = probs2[0].item(), probs2[1].item()
    
    final_probs = {
        '찬성_개정강화': prob_positive,
        '찬성_폐지완화': prob_negative * prob_repeal,
        '반대_현상유지': prob_negative * prob_maintain,
        '중립': prob_neutral
    }

    total_prob = sum(final_probs.values())
    if total_prob > 0:
        final_probs = {k: v / total_prob for k, v in final_probs.items()}

    final_label = max(final_probs, key=final_probs.get)
    return final_label, final_probs
    
def main():
    print(f"Using device: {device}")

    # 모델 로드
    try:
        model1 = CustomModelWithAttention(MODEL_NAME, num_labels=3)
        state_dict1 = load_file(BEST1_MODEL_PATH, device="cpu")
        model1.load_state_dict(state_dict1)
        model1.to(device)
        model1.eval()

        model2 = CustomModelWithAttention(MODEL_NAME, num_labels=2)
        state_dict2 = load_file(BEST2_MODEL_PATH, device="cpu")
        model2.load_state_dict(state_dict2)
        model2.to(device)
        model2.eval()
        print("\nBest models loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    # 테스트 파일 로드 및 예측
    try:
        df_test = pd.read_csv(TEST_FILE_PATH, encoding='utf-8-sig')
    except Exception as e:
        print(f"Test file loading failed: {e}")
        return

    predictions = []
    probabilities = []
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Testing..."):
        pred_label, probs = predict_two_stage(row['content'], model1, model2, tokenizer, device)
        predictions.append(pred_label)
        probabilities.append(probs)

    df_test['predicted_label'] = predictions
    
    # 평가
    y_true = df_test['label'].str.strip("[]'")
    y_pred = df_test['predicted_label']
    
    print("\n==================== FINAL EVALUATION REPORT ====================")
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(report)
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    # 혼동 행렬 출력
    labels_order = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels_order], columns=[f"Pred_{l}" for l in labels_order])
    print("\n--- Confusion Matrix ---")
    print(cm_df)
    print("=================================================================\n")

    # 결과 저장 (선택 사항)
    output_filename = PROCESSED_PATH / "Social_EMO_Test_Results.csv"
    df_test.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"Detailed test results saved to {output_filename}")


if __name__ == "__main__":
    main()