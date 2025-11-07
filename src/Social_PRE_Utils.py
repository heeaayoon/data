# Social_PRE_Utils.py
# 전처리 함수 모듈
import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from typing import Set, Tuple
from tqdm import tqdm
from pathlib import Path

tqdm.pandas()

def save_processed_data(cleaned_df: pd.DataFrame, ads_df: pd.DataFrame, output_dir: Path):
    """결과 데이터프레임을 CSV 파일로 저장합니다."""
    print("\n--- 결과 저장 시작 ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_csv_path = output_dir / "Social_pre.csv"
    removed_csv_path = output_dir / "Social_제거된_광고_데이터.csv"
    
    try:
        cleaned_df.to_csv(cleaned_csv_path, index=False, encoding="utf-8-sig")
        print(f"전처리 완료된 데이터 저장 성공: '{cleaned_csv_path}'")
        
        ads_df.to_csv(removed_csv_path, index=False, encoding="utf-8-sig")
        print(f"제거된 광고 데이터 저장 성공: '{removed_csv_path}'")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")

# 1단계: 데이터 로딩/변환 및 저장 관련 함수
def merge_and_load_excel_files(raw_data_dir: Path, keyword: str, merged_csv_path: Path) -> pd.DataFrame:
    """
    엑셀 파일들의 '모든 시트'를 읽어 병합하고, 병합된 CSV를 데이터프레임으로 로드하여 반환합니다.
    """
    print(f"\n--- 1단계: '{keyword}' 키워드가 포함된 엑셀 파일 병합 및 로딩 ---")
    
    try:
        excel_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".xlsx") and keyword in f]
        
        if not excel_files:
            print(f"-> 병합할 엑셀 파일 없음. 기존 '{merged_csv_path.name}' 파일 로드를 시도합니다.")
        else:
            print(f"-> 대상 파일: {excel_files}")
            df_list = []
            
            for file in excel_files:
                file_path = raw_data_dir / file
                print(f"  -> 파일 읽는 중: {file}")
                try:
                    # sheet_name=None 으로 설정하여 모든 시트를 불러옵니다. (결과는 딕셔너리 형태)
                    all_sheets_dict = pd.read_excel(file_path, sheet_name=None)
                    
                    # 각 시트를 순회하며 df_list에 추가합니다.
                    for sheet_name, sheet_df in all_sheets_dict.items():
                        print(f"    - 시트 '{sheet_name}' 처리 중... ({len(sheet_df)}개 행)")
                        df_list.append(sheet_df)
                        
                except Exception as e:
                    print(f"'{file}' 파일의 시트를 읽는 중 오류 발생: {e}")
                    continue # 문제가 있는 파일은 건너뛰고 계속 진행

            if not df_list:
                print("처리할 데이터 시트가 하나도 없습니다. 병합을 중단합니다.")
                return None

            merged_df = pd.concat(df_list, ignore_index=True)
            merged_df.to_csv(merged_csv_path, index=False, encoding="utf-8-sig")
            print(f"-> {len(excel_files)}개 엑셀 파일의 모든 시트를 '{merged_csv_path.name}'으로 병합 완료. (총 {len(merged_df)}개 행)")
            return merged_df
        
        # 병합할 파일이 없을 경우, 기존 파일 로드
        if merged_csv_path.exists():
            try:
                return pd.read_csv(merged_csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(merged_csv_path, encoding='cp949')
        else:
            print(f"병합할 엑셀 파일도, 기존에 병합된 '{merged_csv_path.name}' 파일도 없습니다.")
            return None

    except Exception as e:
        print(f"엑셀 파일 병합 또는 로딩 중 오류 발생: {e}")
        return None

# 2단계: 텍스트 정제 관련 함수들
def clean_text(text: str) -> str:
    """URL, 이모티콘, 특수문자 제거 및 텍스트 정규화"""
    if pd.isnull(text):
        return ""
    # URL 제거
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', str(text))
    # 이모티콘 제거
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # 특수문자 제거 (한글, 영어, 숫자, 공백 제외)
    text = re.sub(r'[^\w\s가-힣]', '', text)
    # 연속 공백을 하나로
    text = re.sub(r'\s+', ' ', text).strip()
    # 반복되는 자음/모음 처리 (ㅋㅋㅋ -> ㅋㅋ)
    text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1{2,}', r'\1\1', text)
    # 소문자 변환
    text = text.lower()
    return text

def normalize_text_for_blacklist(text: str) -> str:
    """블랙리스트 비교를 위한 텍스트 정규화 (특수문자를 공백으로 처리)"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub(r'[^\w\s가-힣]', ' ', text) # 특수문자를 공백으로 변경
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def create_blacklist_from_content(content_series: pd.Series, min_freq: int, min_len: int) -> Set[str]:
    """본문(content) 시리즈로부터 템플릿 문장 블랙리스트를 생성합니다."""
    print("--- df['content']를 기반으로 블랙리스트 생성을 시작합니다. ---")
    
    # 문장 분리 (줄바꿈, 마침표 등 다양한 기준으로)
    sentences = content_series.astype(str).str.replace(r'_x000d_', '\n').str.split(r'[.\n!?|;:]|ex\s').explode()
    
    # 정규화 및 필터링
    normalized_sentences = sentences.apply(normalize_text_for_blacklist)
    filtered_sentences = normalized_sentences[normalized_sentences.str.len() >= min_len]
    
    # 빈도수 계산 및 블랙리스트 생성
    sentence_counts = Counter(filtered_sentences)
    blacklist_set = {sentence for sentence, count in sentence_counts.items() if count >= min_freq}
    
    print(f"블랙리스트 생성 완료: {len(blacklist_set)}개의 템플릿 문장을 찾았습니다.")
    return blacklist_set

def remove_templates(original_text: str, blacklist: Set[str]) -> str:
    """주어진 텍스트에서 블랙리스트에 포함된 문장을 제거합니다."""
    if not isinstance(original_text, str):
        return ""
        
    raw_sentences = re.split(r'([.\n!?|;:]|ex\s)', original_text) # 구분자도 보존
    
    clean_parts = []
    for i in range(0, len(raw_sentences), 2):
        sentence_part = raw_sentences[i]
        delimiter = raw_sentences[i+1] if i+1 < len(raw_sentences) else ""
        
        normalized_sentence = normalize_text_for_blacklist(sentence_part)
        if normalized_sentence not in blacklist:
            clean_parts.append(sentence_part.strip() + delimiter)

    return " ".join(filter(None, clean_parts)).strip()

# 3단계: 불필요 데이터 제거
def remove_unnecessary_data(df: pd.DataFrame) -> pd.DataFrame:
    """완전 빈 데이터와 중복 데이터를 제거합니다."""
    # 1. 완전 빈 데이터 제거
    initial_count = len(df)
    df = df[
        ~(df['content'].isna()) &
        (df['content'].str.strip() != "") &
        (df['content'].str.strip().str.lower() != 'nan')
    ].copy()
    print(f"빈 데이터 제거: {initial_count - len(df)}개 행 제거됨.")
    
    # 2. 중복 데이터 제거 (날짜 기준 오름차순 정렬 후, 처음 나온 것 유지)
    initial_count = len(df)
    df = df.sort_values('date').drop_duplicates(subset=['title', 'content'], keep='first')
    print(f"중복 데이터 제거: {initial_count - len(df)}개 행 제거됨.")
    return df

def filter_advertisements(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """강력한 광고 패턴을 사용하여 광고성 게시물을 필터링합니다."""
    STRONG_PATTERNS = [
        # ... (이전 Preprocessing2.py의 STRONG_PATTERNS 리스트 전체 복사) ...
        r"(원고료\s*를\s*받[아았]|협찬\s*받[아았]|광고\s*표기|#?\s?(광고|협찬)|\bAD\b|\bPR\b)",
        r"(바로가기|자세히\s*보기|상세\s*보기|링크\s*클릭|구매\s*하기|신청\s*하기|예약\s*하기|구독\s*하기)",
        r"(smartstore\.naver\.com|linktr\.ee|forms\.gle|bit\.ly|me2\.do|t\.co|shorturl\.at|url\.kr|tinyurl\.com|bitly\.com|naver\.me)",
        r"(장바구니|결제\s*하기|무이자|할부|무료\s*배송|교환|환불)",
        r"(\b\d{1,3}\s?%(\s*할인)?|할인\s*\d{1,2}\s*%|할인\s*행사|특가\s*행사)",
        r"((판매가|할인가|행사가|정가)\s*[\d,]+(?:원|원\s*입니다)|\b[\d,]+원\s*(에\s*판매|구매))",
        r"([0-9]{2,3}-[0-9]{3,4}-[0-9]{4})",
        r"\b01[016789][\s\.-]?\d{3,4}[\s\.-]?\d{4}\b",
        r"(\+82[-\s]?(?:10|1[6-9]|2|[3-6][1-5])[-\s]?\d{3,4}[-\s]?\d{4})",
        r"(대표번호|대표전화|고객센터|ARS)\s*\d{2,4}[-]?\d{3,4}[-]?\d{3,4}",
        r"(open\.kakao\.com|카카오톡\s*채널|카카오\s*채널|오픈채팅|카톡\s*오픈|텔레그램|Telegram|라인\s*ID|카카오\s*ID|카톡\s*ID|네이버\s*톡톡|톡톡|플러스친구|플친)",
        r"(분양\s*문의|청약\s*문의|상담전화|상담\s*예약|모델하우스\s*방문\s*예약|견본주택\s*방문)",
        r"(선착순|마감\s*(임박|주의)|한정\s*(수량|판매)|오늘\s*마감|지금\s*신청)",
        r"(수강생\s*모집|원서\s*접수|합격\s*보장|설명회\s*신청)",
        r"(쿠팡|coupang|스마트스토어|스토어찜|오늘의딜|쇼핑라이브|라이브\s*커머스|라방|선물하기|카카오\s*선물|톡딜|네이버페이|카카오페이|토스|페이코)",
        r"(단체\s*주문|대량\s*주문|견적\s*요청|납품|B2B|총판|대리점|입점문의|제휴\s*문의)",
        r"(계좌|입금|입금자명|송금|무통장\s*입금)\s*[^\n]{0,12}\d{2,4}[- ]?\d{3,4}[- ]?\d{3,4}",
        r"(오픈\s*기념|얼리버드|사전\s*(구매|등록|신청))",
        r"(댓글\s*이벤트|팔로우\s*이벤트|공유\s*이벤트|리그램|추첨|당첨|경품)",
    ]
    strong_rgx = re.compile("|".join(STRONG_PATTERNS), re.IGNORECASE)
    
    print("광고 필터링 진행중...")
    is_ad_mask = df['content'].astype(str).progress_apply(lambda x: bool(strong_rgx.search(x)))
    
    df_cleaned = df[~is_ad_mask].copy()
    df_removed_ads = df[is_ad_mask].copy()
    
    print(f"광고 필터링 완료: {len(df_removed_ads)}개 행 제거됨.")
    print(f"최종 데이터: {len(df_cleaned)}개 행")
    
    return df_cleaned, df_removed_ads