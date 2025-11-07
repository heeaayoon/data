# Comment_CRAWL_Run.py
from pathlib import Path
import pandas as pd
from urllib.parse import quote_plus
import time
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta

# utils 모듈에서 필요한 모든 기능 함수들을 가져옵니다.
from Comment_CRAWL_Utils import (
    setup_driver,
    save_progress,
    load_progress,
    crawl_article_content,
    crawl_comments,
    save_to_csv,
    finalize_results
)

# 1. 파일 경로 및 이름 설정
DATA_DIR = "data"
PROCESSED_DIR = "processed"
INPUT_FILENAME = "news_cluster_w_event_name.csv"
FINAL_OUTPUT_FILENAME = "Comment_all.csv"
TEMP_OUTPUT_DIR = "crawled_temp_output"
PROGRESS_FILENAME = "crawling_progress.json"

# 2. 크롤링 기간 설정
START_DATE = (datetime.now() - timedelta(days=2)).strftime("%Y.%m.%d") # 2일전에서
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y.%m.%d") # 어제까지

# 3. 크롤링 세부 설정
MAX_PAGES_PER_QUERY = 10
INTERMEDIATE_SAVE_COUNT = 200

# 4. 수동 재시작 (필요할 때만 사용, 평소에는 None)
MANUAL_RESUME_CATEGORY = None
MANUAL_RESUME_INDEX = None

def main():
    """뉴스 댓글 크롤링 및 후처리 전체 파이프라인을 실행합니다."""
    
    # 1단계: 경로 준비
    base_path = Path(__file__).resolve().parent.parent
    input_path = base_path / DATA_DIR / PROCESSED_DIR / INPUT_FILENAME
    temp_output_dir = base_path / TEMP_OUTPUT_DIR
    final_output_path = base_path / DATA_DIR / PROCESSED_DIR / FINAL_OUTPUT_FILENAME
    progress_file = temp_output_dir / PROGRESS_FILENAME
    
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_path)
        categories = df['predicted_category'].dropna().unique()
    except (FileNotFoundError, KeyError) as e:
        print(f"입력 파일 오류: {e}"); return

    if MANUAL_RESUME_CATEGORY and MANUAL_RESUME_INDEX is not None:
        save_progress(progress_file, MANUAL_RESUME_CATEGORY, MANUAL_RESUME_INDEX)
        print(f"수동 재시작 지점 설정됨: '{MANUAL_RESUME_CATEGORY}'의 {MANUAL_RESUME_INDEX + 1}번째부터")

    driver = setup_driver()
    if not driver: return
    
    # 2단계: 메인 크롤링 실행
    completed_successfully = False
    try:
        start_category, start_index = load_progress(progress_file)
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        can_start_crawling = not start_category

        for category in categories:
            if not can_start_crawling and category != start_category:
                print(f"⏩ 카테고리 [{category}] 건너뛰기")
                continue
            can_start_crawling = True
            
            output_csv_path = temp_output_dir / f"crawled_{category}.csv"
            queries = df[df['predicted_category'] == category][['cluster_event_name', 'predicted_MAIN_NAME']].drop_duplicates().to_dict('records')
            temp_results = []
            
            current_start_index = start_index if category == start_category else 0
            
            for i in range(current_start_index, len(queries)):
                item = queries[i]
                query, main_name = item['cluster_event_name'], item['predicted_MAIN_NAME']
                print(f"\n▶ [{category}] {i+1}/{len(queries)} - \"{query}\"")

                for page in range(1, MAX_PAGES_PER_QUERY + 1):
                    url = f"https://search.naver.com/search.naver?where=news&query={quote_plus(query)}&start={(page-1)*10+1}&pd=3&ds={START_DATE}&de={END_DATE}"
                    try:
                        response = requests.get(url, headers=headers)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        news_containers = [h.parent for h in soup.select('div[data-sds-comp="Profile"]') if h.parent]
                        if not news_containers: break

                        for container in news_containers:
                            title_tag = container.select_one('a[data-heatmap-target=".tit"]')
                            naver_news_tag = container.select_one('a[data-heatmap-target=".nav"]')
                            if not (title_tag and naver_news_tag and "n.news.naver.com" in naver_news_tag['href']): continue
                            
                            news_url = naver_news_tag['href']
                            title = title_tag.get_text(strip=True)
                            content = crawl_article_content(news_url, headers)
                            comments = crawl_comments(driver, news_url)
                            print(f"  - 기사: {title[:30]}... (댓글: {len(comments)}개)")
                            
                            for comment in comments:
                                temp_results.append({
                                    'category': category, 'main_name': main_name, 'query': query,
                                    'title': title, 'url': news_url, 'content': content,
                                    'comment_content': comment['comment_content'], 'comment_date': comment['comment_date']
                                })
                            
                            if len(temp_results) >= INTERMEDIATE_SAVE_COUNT:
                                save_to_csv(temp_results, output_csv_path); temp_results.clear()
                            time.sleep(0.5)
                    except Exception as e: print(f" 페이지 처리 오류: {e}")
                    time.sleep(1)
                
                save_progress(progress_file, category, i)
            
            save_to_csv(temp_results, output_csv_path)
            # 다음 카테고리를 위해 start_index 초기화
            start_index = 0
        
        completed_successfully = True
    finally:
        if driver: driver.quit()
        # 성공적으로 끝나면 progress 파일 삭제
        if completed_successfully and progress_file.exists(): os.remove(progress_file)

    # 3단계: 결과 파일 정리
    finalize_results(temp_output_dir, final_output_path)
    
    print("\n모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()