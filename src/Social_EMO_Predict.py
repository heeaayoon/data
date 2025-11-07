import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoTokenizer

# Social_EMO_Utils.py에서 공통 클래스 및 변수 가져오기
from Social_EMO_Utils import CustomModelWithAttention, MODEL_NAME, MAX_LEN

# 기본 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 경로 설정
DATA_FOLDER_NAME = "data"
PROCESSED_FOLDER_NAME = "processed"
INPUT_FILENAME = "Comment_all.csv"  # 분석할 파일의 이름
OUTPUT_FILENAME = "Comment_all_predicted_EMO.csv" # 저장할 파일의 이름
CONTENT_COLUMN = "content"
TEST_ROWS = None # 테스트할 행 수 (전체는 None)

BASE_PATH = Path(__file__).resolve().parent.parent
INPUT_CSV_PATH = BASE_PATH / DATA_FOLDER_NAME / PROCESSED_FOLDER_NAME / INPUT_FILENAME
OUTPUT_CSV_PATH = BASE_PATH / DATA_FOLDER_NAME / PROCESSED_FOLDER_NAME / OUTPUT_FILENAME

MODEL_PATH = BASE_PATH / "models"
BEST1_MODEL_PATH = MODEL_PATH / "stage1_best_model" / "model.safetensors"
BEST2_MODEL_PATH = MODEL_PATH / "stage2_best_model" / "model.safetensors"

# 모델 로드 함수 (이전과 동일)
def load_models():
    """학습된 1, 2단계 모델을 로드하여 반환합니다."""
    try:
        model1 = CustomModelWithAttention(MODEL_NAME, num_labels=3)
        state_dict1 = load_file(str(BEST1_MODEL_PATH), device=str(device))
        model1.load_state_dict(state_dict1)
        model1.to(device)
        model1.eval()

        model2 = CustomModelWithAttention(MODEL_NAME, num_labels=2)
        state_dict2 = load_file(str(BEST2_MODEL_PATH), device=str(device))
        model2.load_state_dict(state_dict2)
        model2.to(device)
        model2.eval()
        
        print("\nBest models loaded successfully!")
        return model1, model2
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None, None

# 예측 함수 (이전과 동일, 안정성 개선)
def predict_sentiment(text: str, model1, model2):
    """단일 텍스트에 대한 감성을 예측합니다."""
    CHUNK_SIZE = MAX_LEN
    OVERLAP = 128
    STRIDE = CHUNK_SIZE - OVERLAP
    
    if not isinstance(text, str):
        text = ""

    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    content_list = [text]
    if len(tokens) > CHUNK_SIZE:
        content_list = [tokenizer.decode(tokens[i:i+CHUNK_SIZE]) for i in range(0, len(tokens), STRIDE)]

    inputs = tokenizer(content_list, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    inputs['article_ids'] = torch.zeros(len(content_list), dtype=torch.long)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits1 = model1(**inputs)
        probs1 = F.softmax(logits1, dim=-1).squeeze(0)
        
        if probs1.dim() == 1:
            prob_positive, prob_negative, prob_neutral = probs1[0].item(), probs1[1].item(), probs1[2].item()
        else:
            prob_positive, prob_negative, prob_neutral = probs1[0, 0].item(), probs1[0, 1].item(), probs1[0, 2].item()

        logits2 = model2(**inputs)
        probs2  = F.softmax(logits2, dim=-1).squeeze(0)
        
        if probs2.dim() == 1:
            prob_repeal, prob_maintain = probs2[0].item(), probs2[1].item()
        else:
            prob_repeal, prob_maintain = probs2[0, 0].item(), probs2[0, 1].item()
    
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

# (수정) main 함수 - argparse 제거
def main():
    print(f"Using device: {device}")
    model1, model2 = load_models()
    if not model1 or not model2:
        return
        
    try:
        # 인코딩 문제 해결 및 테스트 행 개수 적용
        try:
            df = pd.read_csv(INPUT_CSV_PATH, encoding='utf-8-sig', nrows=TEST_ROWS)
        except UnicodeDecodeError:
            df = pd.read_csv(INPUT_CSV_PATH, encoding='cp949', nrows=TEST_ROWS)
        
        if TEST_ROWS:
            print(f"\n--- 테스트 모드: 처음 {TEST_ROWS}개 행만 처리합니다. ---")
        
        print(f"\n- 입력 파일: {INPUT_CSV_PATH}")
        print(f"- 출력 파일: {OUTPUT_CSV_PATH}\n")

        if CONTENT_COLUMN not in df.columns:
            print(f"오류: '{CONTENT_COLUMN}' 컬럼을 찾을 수 없습니다.")
            return
            
    except FileNotFoundError:
        print(f"오류: '{INPUT_CSV_PATH}' 파일을 찾을 수 없습니다.")
        return

    results = []
    for text in tqdm(df[CONTENT_COLUMN].astype(str), desc="CSV 파일 예측 중..."):
        try:
            label, probs = predict_sentiment(text, model1, model2)
            result_row = {'predicted_label': label}
            result_row.update({f'prob_{k}': v for k, v in probs.items()})
            results.append(result_row)
        except Exception:
            results.append({'predicted_label': '예측 실패'})
    
    results_df = pd.DataFrame(results)
    output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    
    output_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n✅ 예측 완료! 결과가 '{OUTPUT_CSV_PATH}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()