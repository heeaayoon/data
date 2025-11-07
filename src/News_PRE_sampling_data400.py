# News_PRE_sampling_data400.py
# 400건(category별 랜덤 100건 추출) 샘플링 후 라벨링 수작업
import pandas as pd

INPUT_FILE_PATH = '../data/processed/news_pre.csv'
OUTPUT_FILE_PATH = '../data/processed/400_ver1.csv'

CATEGORY_COLUMN = 'category'  # 그룹화할 기준 컬럼
N_SAMPLES = 100             # 각 카테고리별로 추출할 기사 수

try:
    df = pd.read_csv(INPUT_FILE_PATH)
    print(f"✅ '{INPUT_FILE_PATH}' 파일을 성공적으로 불러왔습니다.")
    print(f"원본 데이터 크기: {len(df)} 행")
except FileNotFoundError:
    print(f"오류: '{INPUT_FILE_PATH}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

#  카테고리별 랜덤 샘플링
sampled_df = df.groupby(CATEGORY_COLUMN, group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), N_SAMPLES), random_state=42)
)
sampled_df.reset_index(drop=True, inplace=True)

# 각 카테고리별로 몇 건씩 추출되었는지 확인
print("\n--- 각 카테고리별 추출된 기사 수 ---")
print(sampled_df[CATEGORY_COLUMN].value_counts())
print("------------------------------------")
sampled_df = sampled_df[['content']]
sampled_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
print(f"\n샘플링된 데이터가 '{OUTPUT_FILE_PATH}' 파일에 성공적으로 저장되었습니다.")