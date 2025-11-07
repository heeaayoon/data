# Comment_CRAWL_Utils.py
import pandas as pd
import os
import json
from pathlib import Path
import time
from datetime import datetime, timedelta
import re
import csv
import requests
from bs4 import BeautifulSoup
import shutil

# Selenium 관련 라이브러리
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# WebDriver 및 진행상황 관리 ---
def setup_driver():
    """크롤링에 사용할 웹 드라이버를 설정합니다. (헤드리스 모드 해제)"""
    try:
        print(" Selenium WebDriver를 설정합니다...")
        options = Options()
        # options.add_argument("--headless") # 안정성을 위해 실제 브라우저 창을 띄움
        options.add_argument("--start-maximized")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-extensions")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        print("WebDriver 설정 완료. (브라우저 창이 나타납니다)")
        return driver
    except Exception as e:
        print(f"WebDriver 설정 중 오류 발생: {e}")
        return None

def save_progress(progress_file: Path, category: str, query_index: int):
    """현재까지 완료된 진행 상황을 JSON 파일에 저장합니다."""
    progress = {'last_completed_category': category, 'last_completed_query_index': query_index}
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)

def load_progress(progress_file: Path):
    """저장된 진행 상황을 JSON 파일에서 불러옵니다."""
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
            print(f"이전 진행 상황 로드: {progress}")
            return progress.get('last_completed_category'), progress.get('last_completed_query_index', -1)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None, -1

# 실제 크롤링 동작
def format_naver_date(date_str: str) -> str:
    """네이버 뉴스 댓글의 날짜 형식을 'YYYYMMDDHHMMSS' 문자열로 변환합니다."""
    now = datetime.now()
    date_str = str(date_str).strip()
    try:
        if "방금 전" in date_str or "초 전" in date_str: return str(now.strftime('%Y%m%d%H%M%S'))
        if "분 전" in date_str:
            min_ago = int(re.search(r'\d+', date_str).group())
            return str((now - timedelta(minutes=min_ago)).strftime('%Y%m%d%H%M%S'))
        if "시간 전" in date_str:
            hour_ago = int(re.search(r'\d+', date_str).group())
            return str((now - timedelta(hours=hour_ago)).strftime('%Y%m%d%H%M%S'))
        if "어제" in date_str:
            target_time = now - timedelta(days=1)
            if time_match := re.search(r'(\d{2}):(\d{2})', date_str):
                hour, minute = map(int, time_match.groups())
                target_time = target_time.replace(hour=hour, minute=minute, second=0)
            return str(target_time.strftime('%Y%m%d%H%M%S'))
        if re.match(r'\d{4}\.\d{2}\.\d{2}\. \d{2}:\d{2}', date_str):
            return str(datetime.strptime(date_str, '%Y.%m.%d. %H:%M').strftime('%Y%m%d%H%M00'))
        if re.match(r'\d{4}\.\d{2}\.\d{2}\.', date_str):
            return str(datetime.strptime(date_str, '%Y.%m.%d.').strftime('%Y%m%d000000'))
    except (ValueError, AttributeError): pass
    return str(now.strftime('%Y%m%d%H%M%S'))

def crawl_article_content(article_url: str, headers: dict) -> str:
    """기사 본문을 크롤링합니다."""
    try:
        response = requests.get(article_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        content_area = soup.select_one('#dic_area, #articeBody, #newsct_article')
        if content_area:
            for tag in content_area.select('script, style, .reporter_area'): tag.decompose()
            return content_area.get_text(strip=True, separator='\n')
        return ""
    except Exception: return ""

def crawl_comments(driver: webdriver.Chrome, article_url: str) -> list:
    """댓글 전용 URL로 접속하여, 댓글이 있을 때만 '더보기'를 클릭하고 모든 댓글을 크롤링합니다."""
    if not driver: return []
    match = re.search(r'/article/(\d+)/(\d+)', article_url)
    if not match: return []
    
    press_id, article_id = match.groups()
    comment_page_url = f"https://n.news.naver.com/article/comment/{press_id}/{article_id}"
    
    try:
        driver.get(comment_page_url)
        try:
            comment_count_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.u_cbox_count")))
            comment_count = int(re.sub(r'[^0-9]', '', comment_count_element.text))
            if comment_count == 0: return []
        except (TimeoutException, ValueError): return []
            
        click_count = 0
        while True:
            try:
                more_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.u_cbox_btn_more")))
                driver.execute_script("arguments[0].click();", more_button)
                click_count += 1
                print(f"    -> '더보기' ({click_count}회)", end='\r')
                time.sleep(0.5)
            except (NoSuchElementException, TimeoutException):
                if click_count > 0: print("\n    -> '더보기' 완료.")
                break
                
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        contents = soup.select("span.u_cbox_contents")
        dates = soup.select("span.u_cbox_date")
        
        return [{'comment_content': c.get_text(strip=True), 'comment_date': format_naver_date(d.get_text(strip=True))} for c, d in zip(contents, dates)]
    except Exception as e:
        print(f"\n 댓글 수집 오류: {e}")
        return []

# 파일 저장 및 후처리 
def save_to_csv(data_list: list, file_path: Path):
    """수집된 데이터를 CSV 파일에 추가로 저장합니다."""
    if not data_list: return
    write_header = not file_path.exists()
    try:
        with open(file_path, 'a', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['category', 'main_name', 'query', 'title', 'url', 'content', 'comment_content', 'comment_date']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header: writer.writeheader()
            writer.writerows(data_list)
    except Exception as e: print(f"\n파일 저장 오류: {e}")

def finalize_results(crawled_dir: Path, final_output_path: Path):
    """
    중간 저장된 모든 CSV를 합치고, 중복을 제거한 후 최종 파일을 만들고 임시 폴더를 삭제합니다.
    'comment_date' 열은 반드시 문자열로 유지합니다.
    """
    print("\n\n--- [최종 단계] 결과 파일 병합, 중복 제거 및 정리 ---")
    
    csv_files = [f for f in os.listdir(crawled_dir) if f.startswith('crawled_') and f.endswith('.csv')]
    if not csv_files:
        print("-> 병합할 파일이 없습니다.")
        return

    print(f"-> 대상 파일: {csv_files}")
    df_list = []
    for file in csv_files:
        try:
            # 최종 병합을 위해 파일을 다시 읽을 때도 dtype={'comment_date': str} 옵션을 반드시 추가
            df = pd.read_csv(crawled_dir / file, dtype={'comment_date': str})
            df_list.append(df)
        except Exception as e:
            print(f"  -> [오류] '{file}' 파일 읽기 중 문제 발생: {e}")

    if not df_list:
        print("-> [오류] 파일을 읽어오지 못해 병합을 중단합니다.")
        return

    combined_df = pd.concat(df_list, ignore_index=True)
    
    print(f"\n-> 중복 제거 전 데이터 건수: {len(combined_df)}")
    deduplicated_df = combined_df.drop_duplicates()
    print(f"-> 중복 제거 후 데이터 건수: {len(deduplicated_df)}")
    
    # 데이터 타입이 문자열(object)로 잘 유지되었는지 마지막으로 확인
    if 'comment_date' in deduplicated_df.columns:
        print(f"-> 'comment_date' 최종 데이터 타입: {deduplicated_df['comment_date'].dtype}")

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    deduplicated_df.to_csv(final_output_path, index=False, encoding='utf-8-sig')
    print(f"최종 결과 저장 완료: '{final_output_path}'")
    
    try:
        shutil.rmtree(crawled_dir)
        print(f"임시 폴더 '{crawled_dir.name}' 삭제 완료.")
    except Exception as e:
        print(f"임시 폴더 삭제 오류: {e}")