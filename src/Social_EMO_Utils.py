# Social_EMO_Utils.py
# 여러 파일에서 공통으로 사용되는 CustomModelWithAttention 클래스와 기본 설정

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, ElectraModel

# 기본 설정 (공통)
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
MAX_LEN = 400

# 모델 클래스 정의
# 이 클래스는 train, test, predict 시 모델 구조를 불러오기 위해 공통으로 필요합니다.
class CustomModelWithAttention(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        cfg = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2
        )
        self.bert = AutoModel.from_pretrained(model_name, config=cfg)
        self.attention = nn.Sequential(
            nn.Linear(cfg.hidden_size, 512),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Softmax(dim=0)
        )
        self.classifier = nn.Linear(cfg.hidden_size, num_labels)
        self.class_weights = None
        self.label_smoothing = 0.05

    def forward(self, input_ids, attention_mask, article_ids, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        chunk_embeddings = outputs.last_hidden_state[:, 0, :]

        # 배치 내의 글(article) 개수 결정
        if labels is not None:
            num_articles = labels.shape[0]
        else: # 예측 시
            num_articles = article_ids.max().item() + 1
            
        final_article_embeddings = torch.zeros((num_articles, self.bert.config.hidden_size), device=chunk_embeddings.device)

        for i in range(num_articles):
            article_chunk_embeddings = chunk_embeddings[article_ids == i]
            if article_chunk_embeddings.shape[0] == 0:
                continue

            if article_chunk_embeddings.shape[0] == 1:
                final_article_embeddings[i] = article_chunk_embeddings.squeeze(0)
            else:
                weights = self.attention(article_chunk_embeddings)
                weighted_avg_embedding = torch.sum(article_chunk_embeddings * weights, dim=0)
                final_article_embeddings[i] = weighted_avg_embedding

        logits = self.classifier(final_article_embeddings)

        loss = None
        if labels is not None:
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
            loss = loss_fct(logits, labels)

        return (loss, logits) if loss is not None else logits