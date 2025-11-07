# News_CLU_Clustering.py
# 클러스터링 코드
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from tqdm import tqdm
import os

# 경로 설정
INPUT_CSV_PATH = "../data/processed/news_predict_result.csv"
OUTPUT_DIR = "../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "news_cluster.csv")
EMBEDDING_FILE_PATH = os.path.join(OUTPUT_DIR, "full_article_embeddings.npz")

# 하이퍼파라미터
MERGE_THRESHOLD = 0.80
REASSIGN_THRESHOLD = 0.70 

def get_representative_doc(vectors, texts_in_cluster):
    """대표 문장으로 content를 반환"""
    if not texts_in_cluster or len(vectors) == 0: return None
    centroid = np.mean(vectors, axis=0)
    distances = cosine_distances([centroid], vectors)[0]
    return texts_in_cluster[np.argmin(distances)]

def merge_clusters(df_group, group_embeddings, merge_threshold, cluster_col='cluster_id'):
    valid_clusters = df_group[df_group[cluster_col] != -1]
    if valid_clusters.empty or valid_clusters[cluster_col].nunique() < 2:
        return df_group
    
    centroids = {
        cid: np.mean(group_embeddings[valid_clusters[valid_clusters[cluster_col] == cid].index], axis=0)
        for cid in valid_clusters[cluster_col].unique()
    }
    cids, vectors = list(centroids.keys()), np.array(list(centroids.values()))
    sim_matrix = cosine_similarity(vectors)
    
    merge_map = {}
    temp_id_counter = -2
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            if sim_matrix[i, j] >= merge_threshold:
                id1, id2 = cids[i], cids[j]
                g1, g2 = merge_map.get(id1, id1), merge_map.get(id2, id2)
                if g1 != g2:
                    new_id = min(g1, g2) if isinstance(g1, int) and g1 < 0 else temp_id_counter
                    for k, v in merge_map.items():
                        if v in [g1, g2]: merge_map[k] = new_id
                    merge_map.update({id1: new_id, id2: new_id})
                    if new_id == temp_id_counter: temp_id_counter -= 1
    
    if not merge_map: return df_group
    
    df_group[cluster_col] = df_group[cluster_col].apply(lambda x: merge_map.get(x, x) if x != -1 else x)
    merged_ids = sorted(df_group[(df_group[cluster_col] < 0) & (df_group[cluster_col] != -1)][cluster_col].unique())
    
    if merged_ids:
        max_id = df_group[df_group[cluster_col] >= 0][cluster_col].max() if not df_group[df_group[cluster_col] >= 0].empty else -1
        id_remap = {old: max_id + i + 1 for i, old in enumerate(merged_ids)}
        df_group[cluster_col] = df_group[cluster_col].replace(id_remap)
        
    return df_group

# 메인 실행 로직
def main(df):
    if os.path.exists(EMBEDDING_FILE_PATH):
        print(f"▶ 저장된 임베딩 파일 로드: '{EMBEDDING_FILE_PATH}'")
        embeddings = np.load(EMBEDDING_FILE_PATH)['embeddings']
    else:
        print("▶ 임베딩 생성...")
        model = SentenceTransformer("jhgan/ko-sbert-sts")
        embeddings = model.encode(df["content"].fillna("").tolist(), show_progress_bar=True)
        np.savez_compressed(EMBEDDING_FILE_PATH, embeddings=embeddings)
    print("▶ 임베딩 준비 완료.")

    print("\n단계 1: MAIN_NAME별 로컬 클러스터링, 병합, 재할당 시작")
    processed_groups = []
    
    for main_name, group in tqdm(df.groupby("predicted_MAIN_NAME"), desc="MAIN_NAME별 처리"):
        group = group.copy()
        original_indices = group.index.tolist()
        group.reset_index(drop=True, inplace=True)
        
        group_embeddings_slice = embeddings[original_indices]
        
        n_articles = len(group)
        min_cluster_size = max(2, min(int(n_articles * 0.002), 50))
        n_neighbors = max(5, min(min_cluster_size * 2, 50))
        min_samples = max(3, min(int(min_cluster_size * 0.5), 15))
        
        reduced = umap.UMAP(n_neighbors=n_neighbors, n_components=15, metric="cosine", random_state=42).fit_transform(group_embeddings_slice)
        group['cluster_id'] = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(reduced)
        
        group = merge_clusters(group, group_embeddings_slice, MERGE_THRESHOLD)
        
        noise_mask = group['cluster_id'] == -1
        if noise_mask.sum() > 0 and (~noise_mask).sum() > 0:
            valid_clusters = group[~noise_mask]
            centroids = {cid: np.mean(group_embeddings_slice[valid_clusters[valid_clusters['cluster_id'] == cid].index], 0) for cid in valid_clusters['cluster_id'].unique()}
            cids, vectors = list(centroids.keys()), np.array(list(centroids.values()))
            
            for idx in group[noise_mask].index: 
                sims = cosine_similarity(group_embeddings_slice[idx].reshape(1, -1), vectors)[0]
                best = np.argmax(sims)
                if sims[best] >= REASSIGN_THRESHOLD:
                    assigned_cid = cids[best]
                    group.loc[idx, 'cluster_id'] = assigned_cid
        
        group.index = original_indices
        processed_groups.append(group)

    df_processed = pd.concat(processed_groups)
    print("--- 단계 1 완료 ---")

    print("\n단계 2: 최종 ID 및 이름 부여 시작")
    df_processed['cluster_name'] = ""
    valid_mask = df_processed['cluster_id'] != -1
    df_processed.loc[valid_mask, 'cluster_name'] = df_processed[valid_mask].apply(
        lambda row: f"{row['predicted_MAIN_NAME']}_{int(row['cluster_id'])}", axis=1)
    
    noise_mask = df_processed['cluster_id'] == -1
    if noise_mask.sum() > 0:
        noise_numbers = df_processed[noise_mask].groupby('predicted_MAIN_NAME').cumcount() + 1
        df_processed.loc[noise_mask, 'cluster_name'] = df_processed[noise_mask]['predicted_MAIN_NAME'] + '_noise_' + noise_numbers.astype(str)

    unique_cluster_names = sorted(df_processed[valid_mask]['cluster_name'].unique())
    name_to_id_map = {name: i for i, name in enumerate(unique_cluster_names)}
    df_processed['final_cluster_id'] = df_processed['cluster_name'].map(name_to_id_map).fillna(-1).astype(int)
    print("--- 단계 2 완료 ---")
    
    print("\n단계 3: 최종 결과 정리 및 저장 시작")
    final_reps_map = {}
    valid_clusters_final = df_processed[df_processed['final_cluster_id'] != -1]
    if not valid_clusters_final.empty:
         # 대표 문장을 계산할 때 content 사용
         final_reps_map = {
            cname: get_representative_doc(embeddings[g.index], g["content"].tolist()) 
            for cname, g in valid_clusters_final.groupby('cluster_name')
        }
    df_processed['representative'] = df_processed['cluster_name'].map(final_reps_map)
    df_processed.loc[df_processed['representative'].isnull(), 'representative'] = df_processed['content'] 

    final_noise_count = (df_processed['final_cluster_id'] == -1).sum()
    
    df_processed.drop(columns=['cluster_id', 'category'], inplace=True, errors='ignore')
    df_processed.rename(columns={
        'final_cluster_id': 'cluster_id', 
        'predicted_category': 'category',
        'predicted_MAIN_NAME': 'MAIN_NAME'
    }, inplace=True)
    
    df_processed = df_processed.sort_values(by=['MAIN_NAME', 'cluster_id'])
    
    final_cols = ['category', 'MAIN_NAME', 'date', 'cluster_id', 'cluster_name', 'representative', 'title', 'content', 'url']
    other_cols = [col for col in df_processed.columns if col not in final_cols and col != 'index']
    df_processed = df_processed[final_cols + other_cols]

    df_processed.to_csv(FINAL_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    
    print(f"\n▶ 모든 작업이 완료되었습니다.")
    print(f"  - 최종 노이즈: {final_noise_count}개")
    print(f"  - 최종 결과가 '{FINAL_OUTPUT_CSV}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    try:
        df_initial = pd.read_csv(INPUT_CSV_PATH)
        main(df_initial)
    except FileNotFoundError:
        print(f"오류: '{INPUT_CSV_PATH}' 파일을 찾을 수 없습니다.")