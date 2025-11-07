# Social_PRE_Run.py
# 전처리 과정을 순서대로 호출하며, 설정값(파일 경로 등)을 코드 상단에서 쉽게 관리 가능 
import numpy as np
from pathlib import Path
import sys

# 모든 실제 기능은 utils 모듈에서 가져옵니다.
from Social_PRE_Utils import (
    merge_and_load_excel_files,
    save_processed_data,
    clean_text,
    create_blacklist_from_content,
    remove_templates,
    remove_unnecessary_data,
    filter_advertisements,
)

# 설정
RAW_DATA_DIR_NAME = "raw"
EXCEL_KEYWORD = "(SOCIAL)"
MERGED_CSV_FILENAME = "Social_raw.csv"
OUTPUT_DIR_NAME = "processed"

# 원본 엑셀/CSV의 컬럼 이름
ORIGINAL_CONTENT_COLUMN = "본문"
ORIGINAL_TITLE_COLUMN = "제목"
ORIGINAL_DATE_COLUMN = "날짜"
MIN_TEMPLATE_FREQUENCY = 50
MIN_TEMPLATE_LENGTH = 15

def main():
    """전체 전처리 파이프라인을 실행합니다."""
    
    # 경로 설정
    base_path = Path(__file__).resolve().parent.parent
    raw_data_dir = base_path / "data" / RAW_DATA_DIR_NAME
    processed_data_dir = base_path / "data" / OUTPUT_DIR_NAME
    merged_csv_path = processed_data_dir / MERGED_CSV_FILENAME

    # 1. 엑셀 파일 병합 및 데이터 로딩
    df = merge_and_load_excel_files(raw_data_dir, EXCEL_KEYWORD, merged_csv_path)
    if df is None:
        print("데이터 로딩에 실패하여 전처리를 중단합니다.")
        return

    # 컬럼명 표준화
    print("\n--- 컬럼명 표준화 ('본문'->'content', '제목'->'title', '날짜'->'date') ---")
    try:
        df = df.rename(columns={
            ORIGINAL_CONTENT_COLUMN: "content",
            ORIGINAL_TITLE_COLUMN: "title",
            ORIGINAL_DATE_COLUMN: "date"  # '날짜'를 'date'로 변경
        })
        
        # 'date' 컬럼이 제대로 생성되었는지 확인
        if 'date' not in df.columns:
            print("경고: 'date' 컬럼이 없습니다. 중복 제거 시 정렬 기준이 없어 정확도가 떨어질 수 있습니다.")

        print("컬럼명 변경 완료.")
    except KeyError as e:
        print(f"오류: 원본 데이터에 필요한 컬럼({e})이 없습니다. 설정에서 컬럼명을 확인해주세요.")
        sys.exit()

    # 1. 텍스트 정제
    print("\n--- 1단계: 텍스트 정제 시작 ---")
    df['content'] = df['content'].fillna(df['title'])
    df['content_cleaned'] = df['content'].apply(clean_text)
    
    blacklist_set = create_blacklist_from_content(df['content_cleaned'], MIN_TEMPLATE_FREQUENCY, MIN_TEMPLATE_LENGTH)
    
    if 'title' in df.columns:
        df['title'] = df['title'].astype(str).apply(clean_text).apply(lambda x: remove_templates(x, blacklist_set))

    df['content'] = df['content_cleaned'].apply(lambda x: remove_templates(x, blacklist_set))
    df.drop(columns=['content_cleaned'], inplace=True)
    print("--- 1단계: 텍스트 정제 완료 ---")

    # 2. 불필요 데이터 제거
    print("\n--- 2단계: 불필요 데이터 제거 시작 ---")
    df_processed = remove_unnecessary_data(df)
    df_cleaned, df_removed_ads = filter_advertisements(df_processed)
    print("--- 2단계: 불필요 데이터 제거 완료 ---")

    # 3. 결과 저장
    save_processed_data(df_cleaned, df_removed_ads, processed_data_dir)

    print("\n모든 전처리 과정이 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    main()