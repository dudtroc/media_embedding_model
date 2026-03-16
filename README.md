# Media Embedding Model

BGE-M3 기반 미디어 장면 검색용 임베딩 모델 파인튜닝 프로젝트입니다.
Hard Negative 학습을 통해 유사한 장면 간 구별 능력을 강화합니다.

## 프로젝트 구조

```
media_embedding_model/
├── configs/
│   └── training_config.yaml      # 학습 설정 (하이퍼파라미터, 데이터 생성 옵션)
├── scripts/
│   ├── generate_training_data.py  # GPT API 기반 학습 데이터 생성
│   ├── dataset.py                 # 데이터셋 클래스 & 전처리
│   ├── loss.py                    # 커스텀 손실 함수
│   ├── train.py                   # 모델 파인튜닝
│   ├── evaluate.py                # 검색 기반 평가 (Recall, MRR, NDCG)
│   └── evaluate_metrics.py        # 임계값 기반 정량 평가
├── data/                          # 생성된 학습 데이터 (train/val/test.json)
├── models/                        # 파인튜닝된 모델 체크포인트
└── requirements.txt
```

## 환경 설정

```bash
pip install -r requirements.txt
```

주요 의존성: PyTorch ≥2.0, Transformers ≥4.36, Sentence-Transformers ≥2.3, OpenAI ≥1.0

## 사용법

전체 파이프라인은 **데이터 생성 → 학습 → 평가** 순서로 진행됩니다.

### 1. 학습 데이터 생성

OpenAI GPT API를 사용하여 미디어 장면 + 쿼리 데이터를 생성합니다.
`.env` 파일에 `OPENAI_API_KEY`를 설정해야 합니다.

```bash
# 실시간 생성 (기본)
python scripts/generate_training_data.py --mode realtime

# 배치 API 사용 (50% 비용 절감)
python scripts/generate_training_data.py --mode batch

# 배치 작업 상태 확인
python scripts/generate_training_data.py --mode batch-status

# 배치 결과 다운로드
python scripts/generate_training_data.py --mode batch-download
```

**옵션:**
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | 생성 모드 (realtime/batch/batch-status/batch-download) | realtime |
| `--num_scenes` | 생성할 장면 수 | config 기준 |
| `--genre` | 특정 장르만 생성 | 전체 |

생성된 데이터는 `data/` 디렉토리에 `train.json`, `val.json`, `test.json`으로 분할 저장됩니다.

#### 데이터 형식

각 장면은 다음 구조를 가집니다:

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
    "caption": "장면에 대한 3~5문장의 상세 설명...",
    "Action": ["김서준이 환자의 상태를 확인하고 있다."]
  },
  "confusable_scenes": [{"Place": "...", "caption": "..."}, ...],
  "query": {
    "normal": ["응급실에서 환자를 치료하는 장면"],
    "hard_negative": ["간호사가 환자를 돌보는 장면은?"],
    "negative": ["해변에서 산책하는 장면"]
  }
}
```

### 2. 모델 학습

#### 2-1. Dense Embedding 모델 학습 (기존: BGE-M3)

BGE-M3 모델을 Hard Negative Contrastive Loss로 파인튜닝합니다.

```bash
# 기본 학습 (config 설정 사용)
python scripts/train.py

# 커스텀 설정
python scripts/train.py --config configs/training_config.yaml \
                        --epochs 10 \
                        --batch_size 32 \
                        --learning_rate 2e-5
```

**주요 옵션:**
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--config` | 설정 파일 경로 | configs/training_config.yaml |
| `--epochs` | 학습 에폭 수 | 5 |
| `--batch_size` | 배치 크기 | 16 |
| `--learning_rate` | 학습률 | 1e-5 |
| `--resume` | 체크포인트에서 재개 | - |

**학습 출력물:**
```
models/bge-m3-finetuned/
├── best/                    # 최고 MRR 기준 베스트 체크포인트
├── final/                   # 최종 에폭 모델
├── checkpoint-epoch-{N}/    # 에폭별 체크포인트
└── training_log.json        # 학습 이력 (loss, metrics)
```

**손실 함수 구성:**
- **InfoNCE Loss**: In-batch negatives를 활용한 대조 학습
- **Triplet Margin Loss**: Positive-Negative 간 최소 마진(0.3) 보장
- Hard Negative 가중치 3.0x, Easy Negative 가중치 1.0x

### 2-2. Reranker 모델 학습 (신규: bge-reranker-v2-m3)

Dense embedding(dual-encoder) 대신, (질의, 문서) pair를 함께 넣는 cross-encoder reranker를 학습할 수 있습니다.
기본 모델은 `BAAI/bge-reranker-v2-m3` 입니다.

```bash
# (query, passage) binary classification 방식 (기본)
python scripts/train_reranker.py --mode classification \
  --negative_source hard_negative \
  --negatives_per_positive 1

# pairwise ranking 방식 (pos score > neg score)
python scripts/train_reranker.py --mode pairwise \
  --prefer_confusable \
  --num_neg_passages 1
```

**출력 디렉토리(기본):**
- `models/bge-reranker-v2-m3-finetuned/best`
- `models/bge-reranker-v2-m3-finetuned/final`
- `models/bge-reranker-v2-m3-finetuned/checkpoint-epoch-N`

### 3. 모델 평가

두 가지 평가 스크립트를 제공합니다.

#### 3-1. 검색 기반 평가 (evaluate.py)

Recall, MRR, NDCG 등 검색 지표를 측정합니다.

```bash
# 파인튜닝 모델 단독 평가
python scripts/evaluate.py

# 원본 BGE-M3 vs 파인튜닝 모델 비교
python scripts/evaluate.py --compare

# 특정 모델 경로 지정
python scripts/evaluate.py --model_path ./models/bge-m3-finetuned/best
```

**측정 지표:**
- Recall@1, Recall@5, Recall@10
- MRR (Mean Reciprocal Rank)
- NDCG@10
- Hard Negative Discrimination Rate
- Similarity Gap (Positive vs Hard Negative 유사도 차이)

#### 3-2. 임계값 기반 정량 평가 (evaluate_metrics.py)

유사도 임계값(threshold)에 따른 매칭/거부 성능을 측정합니다.

#### 3-3. Reranker 평가 (evaluate_reranker.py)

Reranker는 (query, passage) pair를 점수화하여 후보 passage를 재정렬합니다.
테스트 데이터에서 Recall@K, MRR 및 hard negative discrimination을 측정합니다.

```bash
# reranker 평가 (후보 전체를 점수화: 비용 큼)
python scripts/evaluate_reranker.py --model_path ./models/bge-reranker-v2-m3-finetuned/best

# 후보 passage 수 제한(정답 포함 N개만 평가) - 빠른 평가용
python scripts/evaluate_reranker.py --model_path ./models/bge-reranker-v2-m3-finetuned/best --num_candidates 200
```

```bash
# 단일 threshold로 평가 (기본 0.85)
python scripts/evaluate_metrics.py

# threshold 지정
python scripts/evaluate_metrics.py --threshold 0.8

# threshold 0.5~1.0 일괄 테스트 (sweep)
python scripts/evaluate_metrics.py --sweep

# 원본 vs 파인튜닝 비교
python scripts/evaluate_metrics.py --compare

# 비교 + sweep (가장 포괄적인 평가)
python scripts/evaluate_metrics.py --compare --sweep
```

**주요 옵션:**
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model_path` | 모델 경로 | models/bge-m3-finetuned/best |
| `--test_data` | 테스트 데이터 경로 | data/test.json |
| `--threshold` | 유사도 임계값 | 0.85 |
| `--sweep` | 0.5~1.0 구간 일괄 테스트 | false |
| `--compare` | 원본 vs 파인튜닝 비교 | false |
| `--output` | 결과 JSON 저장 경로 | 자동 설정 |

**측정 지표:**
| 지표 | 설명 |
|------|------|
| Positive Rate | 정답 쿼리가 threshold 이상으로 매칭된 비율 |
| Negative Rate | 오답 쿼리가 threshold 미만으로 거부된 비율 |
| Hard Negative Reject Rate | Hard Negative 쿼리의 거부율 |
| Easy Negative Reject Rate | Easy Negative 쿼리의 거부율 |
| Separation Success Rate | Normal 유사도 > Hard Negative 유사도인 쌍의 비율 |
| Average Margin | Normal과 Hard Negative 유사도의 평균 차이 |

**Sweep 출력 예시:**
```
  Threshold    Positive    Negative   HN Reject  Neg Reject  Separation  Avg Margin
                Rate(%)     Rate(%)     Rate(%)     Rate(%)     Rate(%)
  ------------------------------------------------------------------------------------------------
        0.5      98.50       12.30       10.20       18.50       92.30      0.1234
        0.6      95.20       25.60       22.10       35.80       92.30      0.1234
        ...
        1.0       0.00      100.00      100.00      100.00       92.30      0.1234
```

## 학습 설정 (training_config.yaml)

핵심 하이퍼파라미터:

```yaml
training:
  base_model: "BAAI/bge-m3"     # 베이스 모델
  epochs: 5                      # 학습 에폭
  batch_size: 16                 # 배치 크기
  learning_rate: 1.0e-5          # 학습률
  max_seq_length: 512            # 최대 토큰 길이
  fp16: true                     # 혼합 정밀도 학습
  loss:
    type: "hard_negative_contrastive"
    temperature: 0.05            # InfoNCE 온도
    margin: 0.3                  # Triplet 마진
    hard_negative_weight: 3.0    # Hard Negative 가중치
```

## Quick Start

```bash
# 1. 환경 설정
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env

# 2. 데이터 생성
python scripts/generate_training_data.py --mode realtime

# 3. 학습
python scripts/train.py

# 4. 평가
python scripts/evaluate.py --compare
python scripts/evaluate_metrics.py --compare --sweep
```
