import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from pathlib import Path
import re

# 1단계: 모든 개별 이슈의 '원시 점수'를 계산하는 함수
def calculate_raw_metrics(news_df, social_df, period_start_date, period_end_date):
    """
    [명세서 기준 수정 완료]
    - V, G, A: 'period_end_date' 이전의 '전체' 뉴스 데이터를 사용
    - P: 'period_start_date' ~ 'period_end_date' 사이의 '기간 내' 뉴스 데이터만 사용
    - C, B: 'period_start_date' ~ 'period_end_date' 사이의 '기간 내' 소셜 데이터만 사용
    """
    news_full_history = news_df[news_df['timestamp'] <= period_end_date]
    news_in_period = news_df[(news_df['timestamp'] >= period_start_date) & (news_df['timestamp'] <= period_end_date)]
    social_in_period = social_df[(social_df['timestamp'] >= period_start_date) & (social_df['timestamp'] <= period_end_date)]

    if news_full_history.empty and social_in_period.empty:
        return pd.DataFrame()

    all_issue_ids = pd.concat([news_full_history['issue_id'], social_in_period['issue_id']]).unique()
    issue_metrics = {}

    for issue in all_issue_ids:
        # V, G, A: '전체 기간' 뉴스 데이터 기반 계산
        news_issue_history = news_full_history[news_full_history['issue_id'] == issue].copy()
        v_score, g_score, a_score = 0.0, 0.0, 0.0
        if not news_issue_history.empty:
            half_life = 7.0
            news_issue_history['days_ago'] = (period_end_date - news_issue_history['timestamp']).dt.total_seconds() / (24 * 3600)
            news_issue_history['weight'] = np.exp(-np.log(2) * news_issue_history['days_ago'] / half_life)
            v_score = news_issue_history['weight'].sum()

            # 'W' 주기의 시작 날짜로 그룹화 키를 변환하여 FutureWarning 방지
            weekly_volume = news_issue_history.groupby(news_issue_history['timestamp'].dt.to_period('W').apply(lambda p: p.start_time))['weight'].sum().resample('W').sum().fillna(0)
            
            v_week_0 = weekly_volume.iloc[-1] if len(weekly_volume) > 0 else 0
            v_week_1 = weekly_volume.iloc[-2] if len(weekly_volume) > 1 else 0
            v_week_2 = weekly_volume.iloc[-3] if len(weekly_volume) > 2 else 0
            
            g_score = (v_week_0 - v_week_1) / v_week_1 if v_week_1 > 0 else v_week_0
            g_previous = (v_week_1 - v_week_2) / v_week_2 if v_week_2 > 0 else v_week_1
            a_score = g_score - g_previous

        # P: '기간 내' 뉴스 데이터 기반 계산
        news_issue_in_period = news_in_period[news_in_period['issue_id'] == issue]
        p_score = 0.0
        if not news_issue_in_period.empty:
            total_days_in_period = (period_end_date - period_start_date).days + 1
            days_with_mentions = news_issue_in_period['timestamp'].dt.normalize().nunique()
            p_score = days_with_mentions / total_days_in_period if total_days_in_period > 0 else 0

        # C, B: '기간 내' 소셜 데이터 기반 계산
        social_issue_in_period = social_in_period[social_in_period['issue_id'] == issue]
        c_score, b_score = 0.0, 0.0
        if not social_issue_in_period.empty:
            sentiment_counts = social_issue_in_period['label'].value_counts()
            pos_count = sentiment_counts.get('찬성_개정강화', 0) + sentiment_counts.get('찬성_폐지완화', 0)
            neg_count = sentiment_counts.get('반대_현상유지', 0)
            
            total_polarized = pos_count + neg_count
            if total_polarized > 1:
                balance_factor = 1 - abs(pos_count - neg_count) / total_polarized
                c_score = total_polarized * balance_factor
            else:
                c_score = 0
            b_score = total_polarized + neg_count

        issue_metrics[issue] = {'V': v_score, 'P': p_score, 'G': g_score, 'A': a_score, 'C': c_score, 'B': b_score}

    return pd.DataFrame.from_dict(issue_metrics, orient='index')


# 2단계: '법안' 이슈를 그룹화하고 최종 TOP 5를 계산하는 함수
def analyze_legislation_top5(raw_metrics_df, legislation_mapping_df):
    if raw_metrics_df.empty or legislation_mapping_df.empty:
        return pd.DataFrame()

    merged_df = raw_metrics_df.merge(legislation_mapping_df, left_index=True, right_on='issue_id', how='inner')
    
    if merged_df.empty or 'legislation_id' not in merged_df.columns:
        return pd.DataFrame()
        
    agg_logic = {
        'V': 'sum', 'P': 'max', 'G': 'mean',
        'A': 'mean', 'C': 'sum', 'B': 'sum'
    }
    aggregated_metrics = merged_df.groupby('legislation_id').agg(agg_logic)

    if len(aggregated_metrics) == 0:
        return pd.DataFrame()

    scaler = MinMaxScaler()
    normalized_metrics = scaler.fit_transform(aggregated_metrics) if len(aggregated_metrics) > 1 else np.full_like(aggregated_metrics, 0.5, dtype=float)

    normalized_df = pd.DataFrame(normalized_metrics, columns=aggregated_metrics.columns, index=aggregated_metrics.index)
    
    weights = {'V': 0.2, 'P': 0.1, 'G': 0.15, 'C': 0.25, 'B': 0.2, 'A': 0.1}
    normalized_df['IIS'] = sum(normalized_df[col] * weights[col] for col in weights)
    
    def assign_grade(score):
        if score >= 0.7: return 'High'
        elif score >= 0.4: return 'Medium'
        else: return 'Low'
    normalized_df['Grade'] = normalized_df['IIS'].apply(assign_grade)

    return normalized_df.sort_values(by='IIS', ascending=False).head(5)

if __name__ == "__main__":
    # 1. 파일 경로 설정
    try:
        base_path = Path(__file__).resolve().parent.parent
    except NameError:
        base_path = Path.cwd()

    processed_path = base_path / "data" / "processed"

    news_data_file = processed_path / "Law_RAG_Result.csv"
    social_data_file = processed_path / "Comment_all_predicted_EMO.csv"
    legal_text_file = processed_path / "Law_total.csv"

    print(f"뉴스 데이터 경로: {news_data_file}")
    print(f"소셜 데이터 경로: {social_data_file}")
    print(f"법률 원문 경로: {legal_text_file}")
    
    try:
        print(f"\n'{news_data_file.name}' 파일을 불러오는 중...")
        news_raw_df = pd.read_csv(news_data_file, dtype={'date': str})
        
        print(f"'{social_data_file.name}' 파일을 불러오는 중...")
        social_raw_df = pd.read_csv(social_data_file, dtype={'comment_date': str})
    
    except FileNotFoundError as e:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. ({e})")
        print("경로 설정이 올바른지, 파일명이 정확한지 확인해주세요.")
        exit()

    # 2. 데이터 전처리
    print("\n데이터 전처리를 시작합니다...")
    
    print(f"필터링 전 원본 뉴스 데이터 행 수: {len(news_raw_df)}")
    condition_to_exclude = (news_raw_df['mapped_law'] == '해당 없음') & (news_raw_df['mapped_article'] == '해당 없음')
    news_raw_df = news_raw_df[~condition_to_exclude].copy()
    print(f"'해당 없음' 법안을 제외한 유효 뉴스 데이터 행 수: {len(news_raw_df)}")

    news_df = news_raw_df.rename(columns={'cluster_event_name': 'issue_id', 'date': 'timestamp'})
    format_string = '%Y%m%d%H%M%S'
    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'], format=format_string, errors='coerce')
    news_df.dropna(subset=['timestamp', 'issue_id', 'mapped_law', 'mapped_article'], inplace=True)

    social_df = social_raw_df.rename(columns={'cluster_event_name': 'issue_id', 'comment_date': 'timestamp'})
    social_df['timestamp'] = pd.to_datetime(social_df['timestamp'], format=format_string, errors='coerce')
    social_df.dropna(subset=['timestamp', 'issue_id', 'label'], inplace=True)

    # 3. '법안 식별자' 및 매핑 테이블 생성
    print("\n'법안 식별자' (legislation_id)를 생성합니다...")
    news_df['legislation_id'] = news_df['mapped_law'] + " " + news_df['mapped_article']
    legislation_mapping_df = news_df[['issue_id', 'legislation_id']].drop_duplicates()
    
    print("--- 생성된 법안 매핑 정보 (샘플) ---")
    print(legislation_mapping_df.head())

    # 4. 전체 기간 설정 및 분석 실행
    all_dates = pd.concat([news_df['timestamp'], social_df['timestamp']])
    if all_dates.empty:
        print("분석할 날짜 데이터가 없습니다.")
    else:
        period_start = all_dates.min()
        period_end = all_dates.max()

        print("\n\n" + "="*50)
        print(f"      전체 기간 최종 '법안' TOP 5 (기간: {period_start.date()} ~ {period_end.date()})")
        print("="*50)
        
        raw_metrics = calculate_raw_metrics(news_df, social_df, period_start, period_end)
        final_top_5 = analyze_legislation_top5(raw_metrics, legislation_mapping_df)
        
        if final_top_5.empty:
            print("분석할 '법안' 관련 데이터가 없습니다.")
        else:
            print("--- [요약] TOP 5 순위표 ---")
            print(final_top_5)

            print("--- TOP 5 법안 상세 분석 (관련 사건 및 법률 내용) ---")

            try:
                legal_df = pd.read_csv(legal_text_file, encoding='utf-8-sig')
                legal_df['legislation_id'] = legal_df['법률명'].astype(str) + " " + legal_df['조 번호'].astype(str)
                legal_content_available = True
            except FileNotFoundError:
                print(f"법률 내용 파일('{legal_text_file.name}')을 찾을 수 없어, 법안 내용은 생략됩니다.")
                legal_content_available = False
            except KeyError as e:
                print(f"법률 내용 파일에 필요한 열({e})이 없습니다. 법안 내용은 생략됩니다.")
                legal_content_available = False

            top_5_ids = final_top_5.index.tolist()
            for rank, leg_id in enumerate(top_5_ids, 1):
                print(f"\n--- [ {rank}위 ] {leg_id} ---")
                
                # 해당 법안 ID와 관련된 모든 뉴스 기록을 원본 news_df에서 가져옵니다.
                related_news = news_df[news_df['legislation_id'] == leg_id]
                
                # 'timestamp'를 기준으로 최신순(내림차순)으로 정렬하고, 중복된 'issue_id' 제거
                sorted_events_df = related_news.sort_values(by='timestamp', ascending=False).drop_duplicates(subset=['issue_id'])
                
                # 정렬된 데이터프레임에서 'issue_id' 컬럼만 리스트로 추출
                associated_events = sorted_events_df['issue_id'].tolist()
                
                if associated_events:
                    print("▶ 관련 주요 사건 (최신순):")
                    # 최대 10개의 사건만 출력
                    for event in associated_events[:10]:
                        print(f"  - {event}")
                else:
                    print("▶ 관련 주요 사건: (정보 없음)")

                # 법률 내용 출력
                if legal_content_available:
                    # leg_id에서 '(조 제목)' 부분을 제거하여 검색용 키를 만들기
                    search_key = re.sub(r'\s*\([^)]*\)', '', leg_id).strip()
                    # 수정된 search_key를 사용하여 법안 내용을 검색
                    content_row = legal_df[legal_df['legislation_id'] == search_key]
                    
                    if not content_row.empty:
                        if '조 제목' in content_row.columns and pd.notna(content_row.iloc[0]['조 제목']):
                            print(f"▶ 조제목: {content_row.iloc[0]['조 제목']}")
                        if '조 내용' in content_row.columns and pd.notna(content_row.iloc[0]['조 내용']):
                            print(f"▶ 조내용:\n{content_row.iloc[0]['조 내용']}")
                        else:
                             print("▶ 조내용: (내용 정보 없음)")
                    else:
                        print(f"▶ 조내용: (일치하는 법안 정보를 찾을 수 없음. 검색 키: '{search_key}')")
                
                print("-" * 50)