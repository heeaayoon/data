# Law_PRE_Run.py

import os
import shutil
from pathlib import Path

# utils 모듈에서 필요한 모든 기능 함수들을 가져옵니다.
from Law_PRE_Utils import (
    parse_law_text_to_dict,
    convert_law_dict_to_csv,
    merge_csv_files
)

# 원본 TXT 파일들이 있는 폴더
INPUT_DIR_NAME = "Law"
# 중간 과정 파일(csv)을 저장할 임시 폴더
TEMP_DIR_NAME = "law_temp_output"
# 최종 결과물을 저장할 폴더 및 파일 이름
OUTPUT_DIR_NAME = "processed"
FINAL_OUTPUT_FILENAME = "Law_total.csv"

def main():
    """법률 TXT 파일을 최종 CSV로 변환하는 전체 파이프라인을 실행합니다."""
    
    # 1. 경로 준비
    base_path = Path(__file__).resolve().parent.parent
    input_dir = base_path / "data" / "processed" / INPUT_DIR_NAME
    temp_dir = base_path / TEMP_DIR_NAME
    output_path = base_path / "data" / "processed" / FINAL_OUTPUT_FILENAME
    
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"입력 폴더: {input_dir}")
    print(f"임시 폴더: {temp_dir}")
    print(f"최종 출력 파일: {output_path}")

    # 2. TXT 파일 목록 가져오기
    try:
        txt_files = list(input_dir.glob("*.txt"))
        if not txt_files: print(f"❗️'{input_dir}'에 .txt 파일이 없습니다."); return
        print(f"\n총 {len(txt_files)}개의 TXT 파일을 처리합니다.")
    except FileNotFoundError: print(f"❗️ 입력 폴더를 찾을 수 없습니다: {input_dir}"); return
    
    intermediate_csv_paths = []

    # 3. 각 TXT 파일을 개별 CSV로 변환
    for txt_path in txt_files:
        print(f"\n--- '{txt_path.name}' 처리 중 ---")
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        law_dictionary = parse_law_text_to_dict(text_content)
        if not law_dictionary: print(f"-> 파싱 실패."); continue

        csv_filename = txt_path.with_suffix(".csv").name
        intermediate_csv_path = temp_dir / csv_filename
        convert_law_dict_to_csv(law_dictionary, intermediate_csv_path)
        intermediate_csv_paths.append(intermediate_csv_path)

    # 4. 개별 CSV 파일들 최종 병합
    merge_csv_files(intermediate_csv_paths, output_path)

    # 5. 임시 폴더 삭제
    try:
        shutil.rmtree(temp_dir)
        print(f"임시 폴더 '{temp_dir.name}' 삭제 완료.")
    except Exception as e: print(f"임시 폴더 삭제 오류: {e}")
        
    print("\n모든 법률 파일 처리 과정이 완료되었습니다.")

if __name__ == "__main__":
    main()