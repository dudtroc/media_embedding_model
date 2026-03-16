# 멀티 GPU 학습 실행 방법 (Accelerate)

이 문서는 본 프로젝트의 `scripts/train.py`(Dense embedding) 및 `scripts/train_reranker.py`(Reranker)를
`accelerate`로 멀티 GPU(DDP) 학습하는 방법을 정리합니다.

## 1) accelerate 설치가 필요한가?

결론: **별도 설치가 필요하지 않은 경우가 대부분**입니다.

- 이 프로젝트의 `requirements.txt`에 이미 `accelerate>=0.25.0`가 포함되어 있습니다.
- 따라서 아래처럼 의존성을 설치했다면 `accelerate`도 함께 설치됩니다.

```bash
pip install -r requirements.txt
```

만약 `accelerate` 명령이 없다고 나오면(예: `command not found`), 아래로 개별 설치할 수 있습니다.

```bash
pip install "accelerate>=0.25.0"
```

## 2) 사전 준비

### (1) GPU 확인

```bash
nvidia-smi
```

### (2) accelerate 설정 파일 생성(최초 1회 권장)

```bash
accelerate config
```

- 프롬프트에서 멀티 GPU 사용 여부, mixed precision(fp16/bf16) 등을 선택합니다.
- 설정 파일은 보통 `~/.cache/huggingface/accelerate/default_config.yaml`에 저장됩니다.

## 3) 실행 방법

> 아래 명령은 **프로젝트 루트 디렉토리**에서 실행하는 것을 기준으로 합니다.

### A. Dense Embedding 모델 학습 (BGE-M3) — `scripts/train.py`

#### 단일 GPU/CPU 실행

```bash
python scripts/train.py --config configs/training_config.yaml
```

#### 멀티 GPU 실행

```bash
accelerate launch scripts/train.py --config configs/training_config.yaml
```

특정 GPU 개수로 실행하고 싶다면:

```bash
accelerate launch --num_processes 2 scripts/train.py --config configs/training_config.yaml
```

### B. Reranker 모델 학습 (bge-reranker-v2-m3) — `scripts/train_reranker.py`

#### 단일 GPU/CPU 실행

```bash
python scripts/train_reranker.py --mode classification
```

#### 멀티 GPU 실행

```bash
accelerate launch scripts/train_reranker.py --mode classification
```

pairwise 모드 멀티 GPU 실행:

```bash
accelerate launch scripts/train_reranker.py --mode pairwise --prefer_confusable
```

특정 GPU 개수로 실행:

```bash
accelerate launch --num_processes 2 scripts/train_reranker.py --mode classification
```

## 4) 출력물 위치

### Dense embedding 학습 결과

- 기본 출력: `models/bge-m3-finetuned/`
  - `best/`, `final/`, `checkpoint-epoch-N/`, `training_log.json`

### Reranker 학습 결과

- 기본 출력: `models/bge-reranker-v2-m3-finetuned/`
  - `best/`, `final/`, `checkpoint-epoch-N/`, `training_log.json`

## 5) 자주 발생하는 이슈

### Q1. `accelerate: command not found`

- 가상환경이 활성화되어 있는지 확인하세요.
- `pip install -r requirements.txt`를 다시 실행하거나, `pip install accelerate`로 개별 설치하세요.

### Q2. 멀티 GPU인데 로그가 여러 번 찍혀요

- 정상입니다. 다만 본 프로젝트의 학습 스크립트는 **main process만 출력/저장**하도록 처리되어 있어
  일반적으로 중복 로그가 최소화됩니다.

### Q3. VRAM 부족(OOM)

- `batch_size`를 줄이거나
- `gradient_accumulation_steps`를 늘리거나
- reranker는 `max_seq_length`를 줄이는 것이 효과적입니다.
