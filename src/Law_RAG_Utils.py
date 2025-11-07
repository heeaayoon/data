# Law_RAG_Utils.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import ollama
import traceback

# --- 프롬프트 정의 ---
QUERY_EXPANSION_PROMPT = """
당신은 주어진 뉴스 기사 [문단]을 분석하여, 관련 법 조항을 찾는 데 가장 효과적인 두 가지 형태의 검색어를 생성하는 최고의 법률 분석 전문가입니다.
[임무]
1.  **'가상의 법률 조항' 생성**: 아래 [법 조항 예시]의 간결하고 명확한 문체와 구조를 **참고**하여, [문단] 내용의 핵심적인 법적 쟁점(위반 행위 또는 의무)을 담은 **새로운** 가상 법률 조항 한 문장을 생성합니다.
2.  **'핵심 키워드' 추출**: 해당 법률 쟁점과 가장 직접적으로 관련된 법률 용어 3~4개를 쉼표(,)로 구분하여 추출합니다.
[법 조항 예시]
* (금지 예시) 누구든지 정보통신망에 의하여 처리ㆍ보관 또는 전송되는 타인의 정보를 훼손하거나 타인의 비밀을 침해ㆍ도용 또는 누설하여서는 아니 된다. (정보통신망법 제49조)
* (의무 예시) 개인정보처리자는 개인정보가 분실ㆍ도난ㆍ유출ㆍ위조ㆍ변조 또는 훼손되지 아니하도록 내부 관리계획 수립, 접속기록 보관 등 안전성 확보에 필요한 기술적ㆍ관리적 및 물리적 조치를 하여야 한다. (개인정보 보호법 제29조)
[규칙]
- 가상 법률 조항은 반드시 [문단]의 내용에 기반하여 **새롭게** 작성해야 합니다.
- 실제 법 조항처럼 핵심 요건을 명확하고 간결하게 표현하십시오.
- 아래 JSON 형식으로만 출력해야 하며, 다른 어떤 설명도 추가하지 마십시오.
```json
{{
  "virtual_law_clause": "생성된 가상의 법률 조항 문장",
  "keywords": "추출된, 핵심, 키워드, 목록"
}}```
---
[문단]
{text_chunk}
---
"""

DRAFT_GENERATION_PROMPT = """
당신은 주어진 법률 조항들과 뉴스 기사의 사실관계를 비교 분석하여 **가장 적합한 단 하나의 법률 조항**을 찾아내는 최고의 AI 법률 분석 전문가입니다. **당신의 선택은 반드시 주어진 후보 목록의 '번호'로만 이루어져야 합니다.**
[분석 대상 법률 목록 (DB)]
{db_law_list_placeholder}
[핵심 분석 원칙]
1. **분석 범위 한정**: 기사의 핵심 주제가 [분석 대상 법률 목록]의 범위를 명백히 벗어나면(예: 형법), 즉시 "해당 없음"으로 판단해야 합니다.
2. **선택 제한**: 최종 결론은 반드시 [후보 법률 조항 목록]에 제시된 후보의 '번호'로만 내려야 합니다. `best_candidate_index` 필드에 숫자만 기입하십시오. 모든 후보가 부적합하다면, `is_relevant`를 `false`로 설정하십시오.
3. **환각 금지**: 목록에 없는 법률명이나 조항을 절대 지어내거나 언급해서는 안 됩니다.
---
이제, 아래 [새로운 분석 대상]에 대한 임무를 다음의 "**단계별 사고 과정**"에 따라 엄격하게 수행하십시오.
### 단계별 사고 과정
**1단계: 핵심 주제 식별 및 분석 범위 확인**
- 이 뉴스 기사의 핵심 법률 쟁점은 무엇입니까? [분석 대상 법률 목록]과 관련이 있습니까? (Yes/No) 관련 없다면 즉시 "해당 없음"으로 결론 내리십시오.
**2단계: 최적 법률 및 조항 탐색**
- 제시된 모든 후보 조항 각각에 대해, 뉴스 기사의 구체적인 사실관계와 조항의 핵심 규정 내용을 철저히 비교 분석하십시오.
- 기사의 핵심 사실관계와 가장 직접적이고 구체적으로 연결되는 단 하나의 조항을 선택하십시오. 모든 후보가 부적합하면 "해당 없음"으로 판단하십시오.
**3단계: 최종 결론 도출 및 근거 명확화**
- `reason` 항목 작성 시: (1)기사의 핵심 쟁점 요약, (2)기사 본문 핵심 구절 '인용', (3)선택한 조항의 핵심 구절 '인용', (4)둘을 연결하는 명확한 논리, (5)다른 주요 후보가 정답이 아닌 이유를 간략히 설명하십시오.
---
[이전 개선 제안]
{last_suggestion}
---
[새로운 분석 대상]
- 뉴스 기사 본문: {news_article}
- **후보 법률 조항 목록 (여기서만 선택해야 함!):** {top_laws}
**오직** 위의 사고 과정과 원칙에 따라 최종 결과를 아래 JSON 형식으로만 출력하십시오.
```json
{{
  "is_relevant": true,
  "best_candidate_index": (1부터 5 사이의 숫자),
  "reason": "[핵심 쟁점 요약] ... [근거 제시] 기사의 '...' 내용은 후보 {best_candidate_index}번 조항의 '...' 규정과 직접 일치합니다. [비교 설명] 반면, 후보 X번은 ...을 다루므로 적합하지 않습니다."
}}
또는 관련이 없는 경우:
{{
  "is_relevant": false,
  "best_candidate_index": null,
  "reason": "[해당 없음 근거] 기사의 핵심 쟁점은 ...이나, 제시된 모든 후보 조항이 각각 ... 와 같은 이유로 직접적인 관련성이 부족하여 '해당 없음'으로 결론 내립니다."
}}
"""

EVALUATION_PROMPT = """
당신은 대한민국 최고의 법률 AI 시스템 성능 평가 전문가입니다. 당신의 유일한 임무는 주어진 [분석 초안]이 아래 [평가 기준]을 얼마나 엄격하게 준수했는지 냉철하게 평가하고, 개선점을 구체적으로 지시하는 것입니다.
[평가 기준]
1.  **근거의 충실성**: `reason`의 모든 내용이 [뉴스 기사]와 [후보 법률 조항 목록]에만 100% 기반합니까?
2.  **논리의 명확성**: 기사 핵심 쟁점, 기사 인용, 법률 조항 인용, 그리고 이 둘을 연결하는 설명이 명확하고 타당합니까?
3.  **선택의 타당성**: 선택된 `best_candidate_index`가 기사의 핵심 사실관계와 가장 직접적으로 연결되는 최적의 선택입니까?
4.  **비교 분석의 구체성**: 다른 후보가 왜 정답이 아닌지에 대한 설명이 구체적이고 설득력이 있습니까?
[임무]
위 [평가 기준]에 따라 [분석 초안]을 평가하고, 반드시 아래 JSON 형식으로만 결과를 출력하십시오.
```json
{{
  "is_perfect": (true 또는 false),
  "score": (10점 만점),
  "critique": "[평가 요약] 근거는 충실하지만, 다른 후보와의 비교 설명이 추상적임.",
  "suggestion_for_refinement": "[개선 지시] '후보 X번은 거리가 멉니다'라는 부분을, 해당 조항의 핵심 단어와 기사 내용의 차이점을 직접 비교하는 방식으로 더 구체적으로 수정하시오."
}}
[평가 대상]
뉴스 기사: {news_article}
후보 법률 조항 목록: {top_laws}
분석 초안: {draft_analysis}
"""

def load_laws_from_csv(csv_file_path):
    all_chunks = []
    try:
        df_law = pd.read_csv(csv_file_path, encoding='utf-8-sig')
        required_columns = ['법률명', '조 번호', '조 제목', '조 내용']
        if not all(col in df_law.columns for col in required_columns):
            print(f"오류: '{csv_file_path}'에 필수 컬럼({', '.join(required_columns)})이 없습니다.")
            return None
    except FileNotFoundError:
        print(f"오류: '{csv_file_path}' 파일을 찾을 수 없습니다."); return None
    
    revision_tag_re = re.compile(r'\s*<개정[^>]+>')
    for _, row in df_law.iterrows():
        law_name = str(row.get('법률명', '')).strip()
        article_number = str(row.get('조 번호', '')).strip()
        article_title = str(row.get('조 제목', '')).strip()
        article_content = str(row.get('조 내용', '')).strip()
        if not article_content: continue
        law_id = f"{article_number} ({article_title})"
        clean_content_full = revision_tag_re.sub('', article_content)
        clean_content_truncated = clean_content_full[:1500]
        text_for_embedding = f"{article_title}. {clean_content_truncated}".strip()
        all_chunks.append({
            "law_name": law_name, "id": law_id,
            "text_for_embedding": text_for_embedding, "content": clean_content_truncated
        })
    return all_chunks

def retrieve_top_k_laws(query, law_chunks, model, corpus_embeddings_cpu, top_k=50):
    if not query: return []
    try:
        query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        query_emb_cpu = query_embedding.cpu().numpy().reshape(1, -1)
        if corpus_embeddings_cpu.ndim == 1:
            corpus_embeddings_cpu = corpus_embeddings_cpu.reshape(1, -1)
        similarities = cosine_similarity(query_emb_cpu, corpus_embeddings_cpu)
        scores = similarities.flatten()
        if not np.all(np.isfinite(scores)):
            scores = np.nan_to_num(scores, nan=-np.inf)
        actual_top_k = min(top_k, len(scores))
        if actual_top_k <= 0: return []
        top_k_indices = np.argsort(scores)[-actual_top_k:][::-1]
        return [(law_chunks[idx], scores[idx]) for idx in top_k_indices if 0 <= idx < len(law_chunks)]
    except Exception as e:
        traceback.print_exc(); return []

def rerank_with_cross_encoder(query, candidates, cross_encoder_model, top_n=20):
    if not candidates: return []
    pairs = [(query, f"{cand.get('law_name', '')} {cand.get('id', '')} {cand.get('content', '')}".strip()) for cand in candidates]
    try:
        scores = cross_encoder_model.predict(pairs, show_progress_bar=False)
        scores = np.array([scores]) if isinstance(scores, (np.float32, float)) else scores
        for i, cand in enumerate(candidates):
            cand['rerank_score'] = scores[i] if i < len(scores) and np.isfinite(scores[i]) else -np.inf
        return sorted(candidates, key=lambda x: x.get('rerank_score', -np.inf), reverse=True)[:top_n]
    except Exception as e:
        traceback.print_exc(); return candidates[:top_n]
    
def expand_and_extract_keywords(text_chunk, llm_model_name):
    prompt = QUERY_EXPANSION_PROMPT.format(text_chunk=text_chunk)
    try:
        response = ollama.chat(model=llm_model_name, messages=[{'role': 'user', 'content': prompt}], format="json")
        result = json.loads(response['message']['content'])
        clause = result.get("virtual_law_clause", "").strip().strip('"\'')
        keywords = result.get("keywords", "").strip()
        return clause, keywords
    except Exception as e:
        print(f"\n쿼리 확장/키워드 추출 오류: {e}"); return text_chunk, ""

def generate_draft_answer(news_article, top_laws, virtual_law_clause, llm_model_name, db_law_list, last_suggestion=""):
    if not top_laws: return None, "관련 법률 조항 후보군이 없습니다."
    context_str = "".join([f"후보 {i+1}: [{law['law_name']}] {law['id']}\n내용: {law['content']}\n\n" for i, law in enumerate(top_laws)])
    law_list_str = "\n".join([f"- {law_name}" for law_name in db_law_list])
    dynamic_prompt = DRAFT_GENERATION_PROMPT.replace("{db_law_list_placeholder}", law_list_str)
    prompt = dynamic_prompt.format(
        news_article=news_article, top_laws=context_str, virtual_law_clause=virtual_law_clause,
        last_suggestion=last_suggestion, best_candidate_index="{best_candidate_index}"
    )
    try:
        response = ollama.chat(model=llm_model_name, messages=[{'role': 'user', 'content': prompt}], format="json")
        return json.loads(response['message']['content']), None
    except Exception as e: return None, f"초안 답변 생성 오류: {e}"

def evaluate_and_refine_answer(news_article, top_laws, draft_analysis_json, llm_model_name):
    context_str = "".join([f"후보 {i+1}: [{law['law_name']}] {law['id']}\n내용: {law['content']}\n\n" for i, law in enumerate(top_laws)])
    prompt = EVALUATION_PROMPT.format(
        news_article=news_article, top_laws=context_str,
        draft_analysis=json.dumps(draft_analysis_json, ensure_ascii=False, indent=2)
    )
    try:
        response = ollama.chat(model=llm_model_name, messages=[{'role': 'user', 'content': prompt}], format="json")
        return json.loads(response['message']['content'])
    except Exception as e:
        print(f"답변 평가 오류: {e}")
        return {"is_perfect": True, "score": 0, "critique": "평가 에이전트 호출 오류", "suggestion_for_refinement": ""}

def format_final_report_from_json(analysis_result, top_laws):
    if not analysis_result: return "분석 결과 없음.", "오류", "오류"
    is_relevant = analysis_result.get('is_relevant', False)
    reason = analysis_result.get('reason', '분석 근거 없음.')
    if not is_relevant:
        return f"## 법률 분석 결과: 해당 없음\n\n**- 분석 근거:**\n{reason}", "해당 없음", "해당 없음"
    selected_index = analysis_result.get('best_candidate_index')
    if isinstance(selected_index, int) and 1 <= selected_index <= len(top_laws):
        selected_law = top_laws[selected_index - 1]
        report = (f"## 법률 분석 결과\n\n"
                  f"**- 관련 법률:** {selected_law.get('law_name', 'N/A')}\n"
                  f"**- 핵심 조항:** {selected_law.get('id', 'N/A')}\n\n"
                  f"**- 분석 근거:**\n{reason}")
        return report, selected_law.get('law_name', 'N/A'), selected_law.get('id', 'N/A')
    else:
        error_msg = f"LLM이 유효하지 않은 후보 번호({selected_index})를 선택."
        report = f"## 법률 분석 결과: 선택 오류\n\n**- 분석 근거:**\n{reason}\n(오류: {error_msg})"
        return report, "선택 오류", "선택 오류"