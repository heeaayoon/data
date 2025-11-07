# News_CLU_Cluster_Summary.py
# 클러스터링 결과 확인용
import pandas as pd
FILE_PATH = "../data/processed/news_cluster.csv"
CLUSTER_ID_COL = "cluster_id"         # 노이즈(-1) 판별을 위해 클러스터 ID 컬럼 추가
CLUSTER_NAME_COL = "cluster_name"
TITLE_COL = "title"
REPRESENTATIVE_COL = "representative" 

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"오류: '{FILE_PATH}' 파일을 찾을 수 없습니다. 파일 경로를 다시 확인해주세요.")
    exit()
except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    exit()

# 필수 컬럼 존재 여부 확인
required_cols = [CLUSTER_ID_COL, CLUSTER_NAME_COL, TITLE_COL, REPRESENTATIVE_COL]
if not all(col in df.columns for col in required_cols):
    print(f"오류: 파일에 '{', '.join(required_cols)}' 컬럼이 모두 존재해야 합니다.")
    print(f"    현재 파일에 있는 컬럼: {df.columns.tolist()}")
    exit()

# 클러스터링 결과 요약 정보 출력
total_articles = len(df)
noise_articles_count = (df[CLUSTER_ID_COL] == -1).sum()
total_valid_clusters = df[df[CLUSTER_ID_COL] != -1][CLUSTER_NAME_COL].nunique()
noise_ratio = (noise_articles_count / total_articles) * 100 if total_articles > 0 else 0

print("="*50)
print("클러스터링 결과 전체 요약")
print(f"총 기사 수\t\t: {total_articles}개")
print(f"생성된 클러스터 수 (노이즈 제외)\t: {total_valid_clusters}개")
print(f"노이즈로 분류된 기사 수\t: {noise_articles_count}개")
print(f"노이즈 비율\t\t: {noise_ratio:.2f}%")
print("="*50)

# MAIN_NAME별 요약 정보 출력
print("\n" + "="*60)
print("MAIN_NAME별 상세 요약")

summary_list = []
for name, group in df.groupby('MAIN_NAME'):
    total = len(group)
    noise = (group[CLUSTER_ID_COL] == -1).sum()
    clusters = group[group[CLUSTER_ID_COL] != -1][CLUSTER_NAME_COL].nunique()
    ratio = (noise / total * 100) if total > 0 else 0
    summary_list.append({
        'MAIN_NAME': name,
        '총 기사 수': total,
        '클러스터 수': clusters,
        '노이즈 수': noise,
        '노이즈 비율 (%)': f"{ratio:.2f}"
    })

if summary_list:
    summary_df = pd.DataFrame(summary_list)
    # 총 기사 수가 많은 순서로 정렬
    summary_df = summary_df.sort_values(by='총 기사 수', ascending=False).reset_index(drop=True)
    print(summary_df.to_string()) 
else:
    print("요약할 데이터가 없습니다.")

print("="*60, "\n")