# Law_RAG_Run.py

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json
import traceback
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

# utils 모듈에서 필요한 모든 기능 함수들을 가져옵니다.
from Law_RAG_Utils import (
    load_laws_from_csv,
    retrieve_top_k_laws,
    rerank_with_cross_encoder,
    expand_and_extract_keywords,
    generate_draft_answer,
    evaluate_and_refine_answer,
    format_final_report_from_json
)

# 설정 
DATA_DIR = "data"
PROCESSED_DIR = "processed"
LAW_FILENAME = "law_total.csv"
NEWS_FILENAME = "news_cluster_w_event_name.csv"
NEWS_CONTENT_COLUMN = "representative"
OUTPUT_FILENAME = "Law_RAG_Result.csv"

LLM_MODEL_NAME = 'command-r'
RETRIEVAL_MODEL_NAME = 'jhgan/ko-sroberta-multitask'
RERANKER_MODEL_NAME = 'bongsoo/klue-cross-encoder-v1'

# NUM_ARTICLES_TO_TEST = 10 # 테스트
NUM_ARTICLES_TO_TEST = None # 전체 실행 시에는 None 으로 설정

def main():
    """RAG 파이프라인 전체를 실행하고, 원본 데이터에 2개 열만 추가합니다."""
    
    # 1. 경로 준비
    base_path = Path(__file__).resolve().parent.parent
    processed_data_dir = base_path / DATA_DIR / PROCESSED_DIR
    law_path = processed_data_dir / LAW_FILENAME
    news_path = processed_data_dir / NEWS_FILENAME
    output_path = processed_data_dir / OUTPUT_FILENAME
    embedding_cache_path = processed_data_dir / 'law_embeddings_1500char.npy'

    # 2. Ollama 서버 연결 확인
    print("\n--- Ollama 서버 연결 확인 중 ---")
    try:
        ollama.list()
        print("Ollama 서버에 성공적으로 연결되었습니다.")
    except Exception as e:
        print("\n[중요] Ollama 서버에 연결할 수 없습니다!")
        print("   - Ollama 애플리케이션이 실행 중인지, 또는 `ollama serve` 명령이 실행 중인지 확인해주세요."); return

    # 3. 데이터 및 모델 로딩
    law_chunks = load_laws_from_csv(law_path)
    if not law_chunks: return
    unique_law_names = sorted(list(set(chunk['law_name'] for chunk in law_chunks)))
    print(f"\n총 {len(law_chunks)}개 법률 조항 로드 완료. ({len(unique_law_names)}개 법률)")
    
    print("\n모델 로드 중...")
    retrieval_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)
    cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)

    # 4. 임베딩 생성 또는 로딩
    corpus_embeddings_cpu = None
    if embedding_cache_path.exists():
        print(f"저장된 법률 임베딩 '{embedding_cache_path.name}' 로드 중...")
        loaded_embeddings = np.load(embedding_cache_path)
        if len(loaded_embeddings) == len(law_chunks):
            corpus_embeddings_cpu = loaded_embeddings
        else: print("임베딩과 법률 데이터 개수가 불일치하여 새로 생성합니다.")
    
    if corpus_embeddings_cpu is None:
        print(f"{len(law_chunks)}개 법률 조항 임베딩 생성 중...")
        corpus_texts = [chunk['text_for_embedding'] for chunk in law_chunks]
        corpus_embeddings = retrieval_model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)
        corpus_embeddings_cpu = corpus_embeddings.cpu().numpy()
        np.save(embedding_cache_path, corpus_embeddings_cpu)
        print(f"새로운 임베딩을 '{embedding_cache_path}'에 저장했습니다.")

    # 5. 뉴스 기사 로딩 및 분석 시작
    try:
        df_news_original = pd.read_csv(news_path, encoding='utf-8-sig')
        if NEWS_CONTENT_COLUMN not in df_news_original.columns:
            print(f"오류: 뉴스 파일에 '{NEWS_CONTENT_COLUMN}' 컬럼이 없습니다."); return
        
        df_to_analyze = df_news_original.dropna(subset=[NEWS_CONTENT_COLUMN]).drop_duplicates(subset=[NEWS_CONTENT_COLUMN]).copy()
        print(f"'{news_path.name}'에서 중복 제외 {len(df_to_analyze)}개 대표 기사 분석 준비 완료.")
    except FileNotFoundError:
        print(f"뉴스 파일 '{news_path}'을(를) 찾을 수 없습니다."); return
    
    analysis_results = []
    temp_output_path = base_path / f"{output_path.stem}_temp.csv"
    num_to_run = len(df_to_analyze) if NUM_ARTICLES_TO_TEST is None else min(NUM_ARTICLES_TO_TEST, len(df_to_analyze))
    
    print(f"\n--- 총 {num_to_run}개 기사에 대해 분석 시작 ---")
    
    for index, row in tqdm(df_to_analyze.head(num_to_run).iterrows(), total=num_to_run, desc="기사 분석 중"):
        content = row[NEWS_CONTENT_COLUMN]
        print(f"\n\n{'='*30} 기사 {index+1} 분석 시작 {'='*30}")
        try:
            virtual_law_clause, keywords = expand_and_extract_keywords(content, LLM_MODEL_NAME)
            
            queries_for_retrieval = [content, virtual_law_clause, keywords]
            aggregated_candidates = {}
            for q in queries_for_retrieval:
                if not q: continue
                results = retrieve_top_k_laws(q, law_chunks, retrieval_model, corpus_embeddings_cpu, top_k=50)
                for cand, score in results:
                    unique_key = f"[{cand['law_name']}] {cand['id']}"
                    if unique_key not in aggregated_candidates or score > aggregated_candidates[unique_key][1]:
                        aggregated_candidates[unique_key] = (cand, score)
            
            sorted_initial_results = sorted(aggregated_candidates.values(), key=lambda item: item[1], reverse=True)
            initial_candidates = [item[0] for item in sorted_initial_results]
            
            reranker_query = f"{virtual_law_clause} {keywords}".strip() or content
            final_candidates = rerank_with_cross_encoder(reranker_query, initial_candidates, cross_encoder, top_n=20)
            
            top_5_laws = final_candidates[:5]
            final_analysis_json = None
            last_suggestion = "최초 분석을 시작합니다."
            for iter_num in range(2):
                print(f"\n--- 🔁 분석 및 개선 시도 ({iter_num + 1}/2) ---")
                draft_analysis_json, error_msg = generate_draft_answer(content, top_5_laws, virtual_law_clause, LLM_MODEL_NAME, unique_law_names, last_suggestion)
                if error_msg:
                    final_analysis_json = {"is_relevant": False, "reason": error_msg}; break
                final_analysis_json = draft_analysis_json
                
                print("평가 에이전트가 생성된 답변을 검토합니다...")
                evaluation = evaluate_and_refine_answer(content, top_5_laws, draft_analysis_json, LLM_MODEL_NAME)
                print(f"평가 점수: {evaluation.get('score', 0)}/10 | 📝 평가 요약: {evaluation.get('critique', 'N/A')}")
                
                if evaluation.get('is_perfect', False) or evaluation.get('score', 0) >= 9:
                    print("평가 결과가 우수하여 분석을 최종 확정합니다."); break
                else:
                    last_suggestion = evaluation.get('suggestion_for_refinement', "")
                    if not last_suggestion or iter_num == 1:
                        print("분석을 최종 확정합니다."); break
                    print(f"개선 지시사항: {last_suggestion}")

            _, mapped_law, mapped_article = format_final_report_from_json(final_analysis_json, top_5_laws)
            
            analysis_results.append({
                NEWS_CONTENT_COLUMN: content,
                "mapped_law": mapped_law,
                "mapped_article": mapped_article,
            })
            
        except Exception as e:
            print(f"기사 처리 중 오류: {e}"); traceback.print_exc()
            analysis_results.append({NEWS_CONTENT_COLUMN: content, "mapped_law": "오류", "mapped_article": f"오류: {e}"})

        if (index + 1) % 10 == 0:
            pd.DataFrame(analysis_results).to_csv(temp_output_path, index=False, encoding='utf-8-sig')

    # 6. 최종 결과 병합 및 저장
    print("\n--- 분석 완료. 원본 데이터와 결과 병합 후 저장 ---")
    try:
        df_analysis = pd.DataFrame(analysis_results)
        final_df = pd.merge(df_news_original, df_analysis, on=NEWS_CONTENT_COLUMN, how='left')
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n최종 결과가 '{output_path}' 파일에 성공적으로 저장되었습니다.")
        if temp_output_path.exists():
            os.remove(temp_output_path)
            print(f"임시 파일 '{temp_output_path.name}'을(를) 삭제했습니다.")
    except Exception as e:
        print(f"\n최종 파일 병합 또는 저장 중 오류: {e}"); traceback.print_exc()

if __name__ == "__main__":
    main()
