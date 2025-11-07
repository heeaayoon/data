# Law_PRE_Utils.py
import pandas as pd
import re
import json
import csv
from pathlib import Path

def parse_law_text_to_dict(text: str) -> dict:
    """
    한국 법률 텍스트를 파싱하여 딕셔너리로 반환합니다.
    (머리글/바닥글 자동 제거, 법률명 추출, <개정> 태그 무시 기능 포함)
    """
    # 머리글/바닥글 자동 제거
    patterns_to_remove = [
        re.compile(r"^\s*법제처\s+\d+\s+국가법령정보센터\s*$", re.MULTILINE),
        re.compile(r"^\s*법제처\s*$", re.MULTILINE),
        re.compile(r"^\s*국가법령정보센터\s*$", re.MULTILINE),
    ]
    cleaned_text = text
    for pattern in patterns_to_remove:
        cleaned_text = pattern.sub("", cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text).strip()
    
    lines = cleaned_text.split('\n')

    law_data = {
        "law_title": "", "enforcement_date": "", "law_number": "",
        "revision_date": "", "competent_authority": "", "contact_info": [],
        "chapters": [], "addendum": []
    }

    # Metadata Parsing
    metadata_lines = []
    start_line_index = 0
    for i, line in enumerate(lines):
        if re.match(r'^\s*제\s*1\s*장', line) or line.strip().startswith("제1조(목적)"):
            start_line_index = i
            break
        metadata_lines.append(line.strip())
    else:
        return None

    if metadata_lines:
        title_candidate = metadata_lines[0]
        if '약칭' in title_candidate:
            law_data["law_title"] = title_candidate
        else:
            law_data["law_title"] = title_candidate.split('[')[0].strip()

    metadata_text = " ".join(metadata_lines)
    metadata_text = re.sub(r'\s+', ' ', metadata_text)

    # Main Content Parsing
    current_chapter = current_section = current_article = current_clause = current_item = current_sub_item = None
    
    chapter_re = re.compile(r'^\s*(제\s*(\d+|[一-龥]+)\s*장)\s+(.*)')
    section_re = re.compile(r'^\s*(제\s*(\d+|[一-龥]+)\s*절)\s+(.*)')
    article_re = re.compile(r'^\s*(제\d+조(?:의\d+)?)\s*(\(.*?\))?(.*)')
    clause_re = re.compile(r'^\s*(①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑪)\s*(.*)')
    item_re = re.compile(r'^[ \t\u00a0]*(\d+(?:의\d+)?)\.\s+(.*)')
    sub_item_re = re.compile(r'^[ \t\u00a0]*([가-힣])\.\s+(.*)')
    tag_re = re.compile(r'\s*<(개정|신설|삭제)\s+([^>]+)>')
    revision_tag_re = re.compile(r'^\s*<개정\s+.*?>\s*$')
    trailing_info_re = re.compile(r'\s*\[(본조신설|전문개정|제목개정)\s+([^\]]+)\]')
    addendum_re = re.compile(r'^\s*부\s*칙\s*(<.*>)?')
    addendum_article_re = re.compile(r'^\s*(제\d+조)\s*(\(.*\))\s*(.*)')
    text_buffer = ""

    def process_buffer(buffer):
        nonlocal current_sub_item, current_item, current_clause, current_article, current_addendum_article, parsing_addendum
        buffer = buffer.strip()
        if not buffer: return ""
        target = None
        if parsing_addendum:
            if current_addendum_article: target = current_addendum_article
        elif current_sub_item: target = current_sub_item
        elif current_item: target = current_item
        elif current_clause: target = current_clause
        elif current_article:
            if not current_article['clauses']:
                current_article['clauses'].append({"clause_number": None, "text": "", "items": []})
            target = current_article['clauses'][-1]
        
        if target:
            clean_buffer = re.sub(r'\s+', ' ', buffer).strip()
            existing_text = target.get('text', '')
            if existing_text and clean_buffer:
                target['text'] = existing_text + " " + clean_buffer
            elif clean_buffer:
                target['text'] = clean_buffer
        return ""

    parsing_addendum = False
    current_addendum = current_addendum_article = None

    for line in lines[start_line_index:]:
        cleaned_line = line.strip()
        if not cleaned_line: continue
        
        if revision_tag_re.match(cleaned_line):
            continue

        addendum_match = addendum_re.match(cleaned_line)
        if addendum_match:
            text_buffer = process_buffer(text_buffer)
            parsing_addendum = True
            addendum_info = addendum_match.group(1) or ""
            current_addendum = {"info": addendum_info.strip(), "articles": []}
            law_data["addendum"].append(current_addendum)
            current_chapter = current_section = current_article = current_clause = current_item = current_sub_item = None
            current_addendum_article = None
            continue

        if parsing_addendum:
            addendum_article_match = addendum_article_re.match(cleaned_line)
            if addendum_article_match:
                text_buffer = process_buffer(text_buffer)
                article_num, article_title, article_text = addendum_article_match.groups()
                current_addendum_article = {"article_number": article_num, "article_title": article_title.strip() if article_title else None, "text": article_text.strip()}
                current_addendum["articles"].append(current_addendum_article)
            elif current_addendum_article:
                text_buffer += " " + cleaned_line if text_buffer else cleaned_line
            continue

        chapter_match = chapter_re.match(cleaned_line)
        section_match = section_re.match(cleaned_line)
        article_match = article_re.match(cleaned_line)
        clause_match = clause_re.match(cleaned_line)
        item_match = item_re.match(cleaned_line)
        sub_item_match = sub_item_re.match(cleaned_line)

        if any([chapter_match, section_match, article_match, clause_match, item_match, sub_item_match]):
            text_buffer = process_buffer(text_buffer)

        if chapter_match:
            current_chapter = {"chapter_number": chapter_match.group(1).strip(), "chapter_title": chapter_match.group(3).strip(), "sections": []}
            law_data["chapters"].append(current_chapter)
            current_section = {"section_number": None, "section_title": None, "articles": []}
            current_chapter["sections"].append(current_section)
            current_article = current_clause = current_item = current_sub_item = None
        elif section_match and current_chapter:
            if current_section and current_section["section_number"] is None and not current_section["articles"]:
                current_chapter["sections"].pop()
            current_section = {"section_number": section_match.group(1).strip(), "section_title": section_match.group(3).strip(), "articles": []}
            current_chapter["sections"].append(current_section)
            current_article = current_clause = current_item = current_sub_item = None
        elif article_match and current_section:
            article_num, article_title, remaining_text = article_match.groups()
            current_article = {"article_number": article_num, "article_title": article_title.strip() if article_title else None, "clauses": []}
            current_section["articles"].append(current_article)
            current_clause = current_item = current_sub_item = None
            text_buffer = remaining_text.strip()
        elif clause_match and current_article:
            current_clause = {"clause_number": clause_match.group(1), "text": clause_match.group(2).strip(), "items": []}
            current_article["clauses"].append(current_clause)
            current_item = current_sub_item = None
        elif item_match and current_article:
            if not current_clause:
                current_clause = {"clause_number": None, "text": "", "items": []}
                current_article["clauses"].append(current_clause)
            current_item = {"item_number": item_match.group(1), "text": item_match.group(2).strip(), "sub_items": []}
            current_clause["items"].append(current_item)
            current_sub_item = None
        elif sub_item_match and current_item:
            current_sub_item = {"sub_item_number": sub_item_match.group(1), "text": sub_item_match.group(2).strip()}
            current_item["sub_items"].append(current_sub_item)
        elif cleaned_line:
            text_buffer += " " + cleaned_line if text_buffer else cleaned_line
    
    process_buffer(text_buffer)
    for chapter in law_data["chapters"]:
        chapter["sections"] = [s for s in chapter["sections"] if s.get("articles") or s.get("section_number")]
    return law_data

def _clean_and_join_text_from_clauses(clauses: list) -> str:
    """조(article) 내부의 모든 텍스트를 정리하고 하나로 합칩니다."""
    full_text = []
    for clause in clauses:
        text_parts = [clause.get('text', '')]
        for item in clause.get('items', []):
            text_parts.append(item.get('text', '')); 
            for sub_item in item.get('sub_items', []): text_parts.append(sub_item.get('text', ''))
        
        clause_full_text = ' '.join(filter(None, text_parts))
        
        temp_cleaned_for_check = re.sub(r'<.*?>', '', clause_full_text, flags=re.DOTALL).strip()
        if temp_cleaned_for_check == '삭제':
            continue

        cleaned_text = re.sub(r'<.*?>|\[.*?\]', '', clause_full_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'[①-⑳]', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if cleaned_text:
            full_text.append(cleaned_text)

    final_content = ' '.join(full_text)
    return re.sub(r'\s+', ' ', final_content).strip()

def convert_law_dict_to_csv(law_data: dict, csv_path: Path):
    """파싱된 법률 딕셔너리를 CSV 파일로 변환하여 저장합니다."""
    law_title = law_data.get('law_title', 'N/A')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        headers = ['법률명', '장 번호', '장 제목', '절 번호', '절 제목', '조 번호', '조 제목', '조 내용']
        writer.writerow(headers)
        for chapter in law_data.get('chapters', []):
            for section in chapter.get('sections', []):
                for article in section.get('articles', []):
                    if not article.get('article_number'): continue
                    content = _clean_and_join_text_from_clauses(article.get('clauses', []))
                    if not content: continue
                    row = [
                        law_title, chapter.get('chapter_number', ''), chapter.get('chapter_title', ''),
                        section.get('section_number', ''), section.get('section_title', ''),
                        article.get('article_number', ''), article.get('article_title', ''), content
                    ]
                    writer.writerow(row)
    print(f"-> '{csv_path.name}' 파일 생성 완료.")

def merge_csv_files(file_paths: list, output_path: Path):
    """주어진 경로의 CSV 파일 목록을 하나로 병합합니다."""
    if not file_paths:
        print("-> 병합할 CSV 파일이 없습니다.")
        return
    try:
        df_list = [pd.read_csv(f, encoding='utf-8') for f in file_paths]
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n최종 병합 완료: {len(file_paths)}개 파일을 '{output_path.name}'으로 저장했습니다.")
    except Exception as e:
        print(f"\n파일 병합 중 오류 발생: {e}")