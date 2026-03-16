# Media Embedding Model - 프로젝트 가이드

BGE-M3 기반 미디어 장면 검색용 임베딩 모델 파인튜닝 프로젝트.
Hard Negative Contrastive Learning을 통해 유사 장면 간 구별 능력을 강화한다.

---

## 프로젝트 구조

```
media_embedding_model/
├── CLAUDE.md                          # 이 파일
├── README.md                          # 사용법 전체 문서
├── .env                               # OPENAI_API_KEY 설정 (git 제외)
├── requirements.txt                   # 패키지 의존성
├── test.py                            # 간단한 테스트 스크립트 (미완성)
├── configs/
│   └── training_config.yaml           # 학습 설정 (하이퍼파라미터, 데이터 생성 옵션)
├── scripts/
│   ├── generate_training_data.py      # GPT API 기반 학습 데이터 생성
│   ├── dataset.py                     # SceneTripletDataset, collate_fn
│   ├── loss.py                        # HardNegativeContrastiveLoss, OnlineHardNegativeMiningLoss
│   ├── train.py                       # BGE-M3 파인튜닝 메인 스크립트
│   ├── evaluate.py                    # Recall/MRR/NDCG 검색 기반 평가
│   ├── evaluate_metrics.py            # Threshold 기반 정량 평가 (원본 vs 파인튜닝)
│   └── evaluate_compare_models.py     # 다중 모델 비교 평가 (BGE-M3, Qwen3-Embedding 등)
├── data/
│   ├── train.json                     # 학습 데이터 (71,887 scenes)
│   ├── val.json                       # 검증 데이터
│   ├── test.json                      # 테스트 데이터 (8,987 scenes)
│   ├── raw_{genre}.json               # 장르별 원시 데이터
│   ├── raw_all.json                   # 전체 통합 원시 데이터
│   ├── scenes/                        # 장르별 장면 데이터 디렉토리
│   └── batch/                         # OpenAI Batch API 요청/응답 파일
└── models/
    ├── bge-m3-finetuned_20000/        # 20000 스텝 학습된 모델
    │   ├── best/                      # 최고 MRR 기준 베스트 체크포인트
    │   └── training_log.json          # 학습 이력
    └── evaluation_results.json        # 평가 결과
```

---

## 데이터 형식

### 학습 데이터 JSON 구조 (각 scene)
```json
{
  "genre": "drama",
  "metadata": {
    "Place": "병원 응급실",
    "Approximate Time": "새벽 2시",
    "Atmosphere": "긴박한",
    "Keywords": ["응급실", "의사", "환자"],
    "Main Characters": [
      {"type": "의사", "name": "김서준", "description": "..."}
    ],
    "caption": "3~5문장 상세 장면 설명",
    "Action": ["행동 묘사 문장들"]
  },
  "confusable_scenes": [{"Place": "...", "caption": "..."}, ...],
  "query": {
    "normal": ["정답 쿼리 3개"],
    "hard_negative": ["혼동 유발 쿼리 5개"],
    "negative": ["명확히 다른 쿼리 2개"]
  }
}
```

### Passage 변환 (`_metadata_to_passage`)
metadata dict → 자연어 문자열 (장소/시간/분위기 + caption + 등장인물 + Action + Keywords)

---

## 핵심 모듈 설명

### scripts/loss.py
- **HardNegativeContrastiveLoss**: InfoNCE + 가중 Triplet Margin Loss 결합
  - temperature=0.05, margin=0.3, hard_negative_weight=3.0, negative_weight=1.0
  - 반환: `{"loss", "infonce_loss", "triplet_loss"}`
- **OnlineHardNegativeMiningLoss**: In-batch에서 가장 어려운 negative 자동 채굴
  - 반환: `{"loss", "infonce_loss", "hard_margin_loss"}`

### scripts/dataset.py
- **SceneTripletDataset**: (query, positive, hard_negatives, negatives) 구성
  - hard_negative: confusable_scenes 기반 + 부족시 동일 장르 보완
  - negative: 전체 passages에서 랜덤 선택
- **collate_fn**: 가변 길이 negative를 flatten 후 `_counts` 배열로 관리

### scripts/train.py
- BGE-M3 CLS 토큰 임베딩 활용 (`last_hidden_state[:, 0, :]`)
- AdamW + linear warmup scheduler
- fp16 혼합 정밀도 (CUDA), gradient accumulation 지원
- 에폭별 체크포인트 저장, best MRR 기준 best 모델 저장
- 출력: `models/bge-m3-finetuned/{best,final,checkpoint-epoch-N}/`

### scripts/evaluate.py
- 검색 기반 평가: Recall@1/5/10, MRR, Hard Negative Discrimination Rate
- `--compare`: 원본 BGE-M3 vs 파인튜닝 모델 비교

### scripts/evaluate_metrics.py
- Threshold 기반 평가: Positive Rate, Negative Rate, Hard Neg Reject Rate, 분리 성공률, 평균 마진
- `--sweep`: 0.5~1.0 구간 0.1 간격 일괄 테스트 (임베딩 한 번만 계산 후 재사용)
- `--compare`: 원본 BGE-M3 vs 파인튜닝 비교

### scripts/evaluate_compare_models.py
- 다중 모델 비교 평가: BGE-M3 (원본/파인튜닝) + 외부 HuggingFace 모델을 동일 테스트셋으로 비교
- 현재 지원 모델: `bge-m3`, `bge-m3-finetuned`, `qwen3` (Qwen/Qwen3-Embedding-0.6B)
- 모델별 인코딩 차이 자동 처리
  - BGE-M3: CLS token pooling, right padding
  - Qwen3-Embedding: last token pooling, left padding, query에 instruction prefix 추가
- `--models`: 평가할 모델 선택 (기본값: 전체 3개)
- `--sweep`: 0.5~1.0 구간 threshold 일괄 테스트
- `MODEL_REGISTRY`에 항목 추가로 새 모델 확장 가능

---

## 학습 하이퍼파라미터 (configs/training_config.yaml)

| 항목 | 값 |
|------|-----|
| base_model | BAAI/bge-m3 |
| epochs | 5 |
| batch_size | 16 |
| learning_rate | 1e-5 |
| warmup_ratio | 0.1 |
| weight_decay | 0.01 |
| max_seq_length | 512 |
| fp16 | true |
| gradient_accumulation_steps | 2 |
| loss type | hard_negative_contrastive |
| temperature | 0.05 |
| margin | 0.3 |
| hard_negative_weight | 3.0 |

데이터 생성 설정: total_samples=90000, train 80%/val 10%/test 10%
장르: drama, movie, documentary, news, variety_show, animation, music_video, advertisement, sports, education

---

## 현재 학습된 모델 성능 (models/bge-m3-finetuned_20000)

`evaluation_results.json` 기준 (evaluate.py 결과):
- Recall@1: 0.7375
- Recall@5: 0.9199
- Recall@10: 0.9666
- MRR: 0.8180
- Hard Negative Discrimination Rate: 0.6263
- Avg Positive Similarity: 0.6019
- Avg Hard Negative Similarity: 0.4077
- Positive-HN Similarity Gap: 0.1942

학습 5 epoch 완료 기준 val metrics (training_log.json):
- val Recall@1: 0.2359 / Recall@5: 0.7701 / MRR: 0.4625 / NDCG@10: 0.5556
- train loss: 0.1130 (infonce: 0.0896 + triplet: 0.0234)

> 참고: evaluation_results.json은 test set 기준이고 training_log.json은 val set 기준이므로 수치 차이가 있음.
> 현재 저장된 모델 디렉토리명이 `bge-m3-finetuned_20000`이며 evaluate_metrics.py의 기본 경로(`bge-m3-finetuned/best`)와 다름 → 평가 시 `--model_path` 명시 필요.

---

## 자주 쓰는 커맨드

```bash
# 데이터 생성 (실시간)
python scripts/generate_training_data.py --mode realtime

# 배치 API 활용 (50% 절감)
python scripts/generate_training_data.py --mode batch
python scripts/generate_training_data.py --mode batch-status
python scripts/generate_training_data.py --mode batch-download

# 학습
python scripts/train.py
python scripts/train.py --epochs 10 --batch_size 32 --learning_rate 2e-5

# 검색 기반 평가
python scripts/evaluate.py --model_path ./models/bge-m3-finetuned_20000/best --compare

# 임계값 기반 평가 (전체)
python scripts/evaluate_metrics.py --model_path ./models/bge-m3-finetuned_20000/best --compare --sweep

# 다중 모델 비교 평가 (BGE-M3 원본 + 파인튜닝 + Qwen3-Embedding)
python scripts/evaluate_compare_models.py --finetuned_path ./models/bge-m3-finetuned_20000/best --sweep

# 특정 모델만 선택해서 비교
python scripts/evaluate_compare_models.py --models bge-m3 qwen3 --sweep

# 단일 threshold로 전체 비교
python scripts/evaluate_compare_models.py --finetuned_path ./models/bge-m3-finetuned_20000/best --threshold 0.8
```

---

## 주의 사항

- `.env`에 `OPENAI_API_KEY` 필요 (데이터 생성 시)
- `scripts/` 디렉토리 내 파일들은 서로 import 관계 있음 (`train.py` → `dataset.py`, `loss.py`)
- 현재 모델 경로: `models/bge-m3-finetuned_20000/best/` (README 기본값 `bge-m3-finetuned/best`와 다름)
- `test.py`는 미완성 파일 (현재 `print("Hello, World!")` 수준)
- GPU 없으면 CPU 자동 fallback (학습은 GPU 권장)
- `metadata_to_passage()` 함수가 `dataset.py`와 `evaluate.py`, `evaluate_metrics.py`, `evaluate_compare_models.py`에 중복 구현되어 있음
