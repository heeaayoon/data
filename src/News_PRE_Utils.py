# News_PRE_Utils.py
# 전처리 함수 모듈
import pandas as pd
import os
from pathlib import Path

def merge_news_excel_files(raw_data_dir: Path, output_csv_path: Path):
    """
    raw 폴더에서 (NEWS) 키워드가 포함된 엑셀 파일들을 찾아
    하나의 CSV 파일('news_raw.csv')로 병합하여 저장합니다.
    """
    print("\n--- [Step 1] (NEWS) 포함 엑셀 파일 병합 시작 ---")
    keyword = "(NEWS)"
    
    try:
        # .xlsx 확장자를 가진 (NEWS) 포함 파일 목록 가져오기
        excel_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".xlsx") and keyword in f]
        
        if not excel_files:
            print(f"-> [알림] '{keyword}' 키워드를 포함하는 엑셀 파일이 없습니다.")
            return

        print(f"-> 대상 파일: {excel_files}")
        df_list = [pd.read_excel(raw_data_dir / file) for file in excel_files]
        
        if not df_list:
            print("-> [알림] 처리할 데이터가 없습니다.")
            return

        merged_df = pd.concat(df_list, ignore_index=True)
        # processed 폴더가 없으면 생성
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"-> [완료] {len(excel_files)}개 파일을 합쳐 '{output_csv_path.name}' 저장. (총 {len(merged_df)}개 행)")

    except Exception as e:
        print(f"-> [오류] 엑셀 병합 중 문제 발생: {e}")


def preprocess_news_data(input_csv_path: Path, output_csv_path: Path):
    """
    병합된 뉴스 데이터('news_raw.csv')를 정제하여 최종 결과 파일('news_pre.csv')로 저장합니다.
    """
    print("\n--- [Step 2] 뉴스 데이터 정제 시작 ---")
    try:
        df = pd.read_csv(input_csv_path, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"-> [오류] 입력 파일 '{input_csv_path.name}'을 찾을 수 없습니다. Step 1을 먼저 실행하세요.")
        return
    except Exception as e:
        print(f"-> [오류] 파일 로딩 중 문제 발생: {e}")
        return

    # 1. content가 비어있거나 NaN인 행 제거
    initial_count = len(df)
    df.dropna(subset=["content"], inplace=True)
    df = df[df["content"].str.strip() != ""]
    print(f"-> content 비어있는 행 제거: {initial_count} → {len(df)}")

    # 2. url + content 기준 중복 제거
    initial_count = len(df)
    df.drop_duplicates(subset=["url", "content"], keep="first", inplace=True)
    print(f"-> url+content 중복 제거 후 행 수: {len(df)}")

    # 3. 불필요한 키워드가 포함된 기사 제거
    keywords = [
        "오늘의 국회일정", "오늘의 주요일정", "미리보는 이데일리 신문",
        "연합뉴스 이 시각 헤드라인", "데일리안 1분뉴스", "헤드라인", "온라인 핫 뉴스만 콕콕"
    ]
    pattern = "|".join(keywords)
    
    initial_count = len(df)
    # content 또는 title에 키워드가 포함된 행을 제거
    df = df[~(df["content"].str.contains(pattern, na=False, regex=True) |
              df["title"].str.contains(pattern, na=False, regex=True))]
    print(f"-> 키워드 포함 기사 제거 후 행 수: {initial_count} → {len(df)}")

    # 4. 최종 파일 저장
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"-> [완료] '{output_csv_path.name}' 저장 완료")