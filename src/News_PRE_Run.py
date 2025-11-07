# News_PRE_Run.py
# 전처리 과정을 순서대로 호출하며, 설정값(파일 경로 등)을 코드 상단에서 쉽게 관리 가능 

from pathlib import Path
from News_PRE_Utils import merge_news_excel_files, preprocess_news_data

def main():
    """
    News 데이터 전처리 전체 파이프라인.
    1. (NEWS) 엑셀 파일들을 하나로 병합하여 'news_raw.csv' 생성.
    2. 'news_raw.csv'를 정제하여 'news_pre.csv' 생성.
    """
    
    # 경로 설정
    base_path = Path(__file__).resolve().parent.parent # AIDA/ 폴더
    
    raw_data_dir = base_path / "data" / "raw"
    processed_data_dir = base_path / "data" / "processed"
    
    # 각 단계별 입/출력 파일 경로
    merged_csv_path = processed_data_dir / "news_raw.csv"  # 1단계 출력, 2단계 입력
    final_csv_path = processed_data_dir / "news_pre.csv"    # 2단계 최종 출력
    
    # --- 전처리 실행 ---
    
    # Step 1: 엑셀 파일 병합
    merge_news_excel_files(raw_data_dir, merged_csv_path)
    
    # Step 2: 병합된 CSV 파일 정제
    preprocess_news_data(merged_csv_path, final_csv_path)
    
    print("\n모든 뉴스 데이터 전처리 과정이 완료되었습니다.")

if __name__ == "__main__":
    main()