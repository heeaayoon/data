# LLM 이름 붙이기
import pandas as pd
import os
from openai import OpenAI

df = pd.read_csv("../../data/processed/news_cluster.csv")
#df = df.sample(n=20, random_state=42) # 테스트용
print(f"총 {len(df)}건 데이터 사용")

# OpenAI 클라이언트 세팅
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# GPT 요청 함수/프롬프트
def generate_event_name(representative: str, model: str = "gpt-4o-mini") -> str:
    if pd.isna(representative) or not str(representative).strip():
        return "대표 기사 없음"

    system_message = (
        "당신은 대표 기사를 보고, 사건을 대표하는 구체적인 '소분류' 이름을 생성하는 전문가입니다. "
        "소분류명은 5~20자 이내의 명사형으로, 사건의 주요 특징, '장소', 대상, 행위 등을 반영해야 합니다. "
        "너무 일반적이거나 포괄적인 단어는 피하고, 설명 없이 단어 또는 짧은 명사구만 출력하세요. "
        "장소가 있다면 반드시 포함하세요"
        "반드시 각 사건명은 고유해야 합니다."
    )

    user_message = (
        f"다음 대표 기사 내용을 바탕으로, 이 클러스터에 적합한 '소분류' 이름을 하나 생성해주세요.\n\n"
        f"## 대표 기사:\n{representative}\n\n"
        "조건:\n"
        "- 5~20자, 명사형 또는 짧은 명사구\n"
        "- 사건의 주요 특징, 대상, 장소, 행위 포함\n"
        "- 설명 없이 단어 또는 짧은 명사구만\n"
        "예시 출력:\n"
        "- 서울 강남 아파트 화재\n"
        "- 부산시 초등학생 학대사건\n"
        "- 의료기관 개인정보 유출\n"
        "- 이하늘 사이버 명예훼손 게시글\n"
        "이 형식에 맞게 생성해주세요:"
    )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=30,
                n=1,
            )
            return response.choices[0].message.content.strip().replace('"', '').replace("'", "")
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return f"[생성 실패: {e}]"


# 클러스터별 대표 기사로 소분류명 생성
cluster_names = {}
print("\n=== 클러스터 소분류명 생성 결과 ===")
for cluster_name, sub in df.groupby("cluster_name"):
    representative = sub["representative"].iloc[0]  # 대표 기사 1개 이용
    event_name = generate_event_name(representative)
    print(f"[{cluster_name}] → {event_name}")
    cluster_names[cluster_name] = event_name

df["cluster_event_name"] = df["cluster_name"].map(cluster_names)
df.to_csv("../../data/processed/news_cluster_w_event_name.csv", index=False, encoding="utf-8-sig")