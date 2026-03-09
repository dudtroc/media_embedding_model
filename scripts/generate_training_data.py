"""
BGE-M3 Hard Negative 개선을 위한 학습 데이터 생성 스크립트.

GPT API를 사용하여 다양한 미디어 장르의 장면 메타데이터와
normal / hard_negative / negative 질의를 생성합니다.

주요 특징:
- 풍부한 캡션 (3~5문장, 다중 인물 행동 서술)
- Confusable Scene (혼동 유발 장면) 기반 Hard Negative passage 생성
- Entity-Attribute Swap 기반 Hard Negative 질의 생성
- 증분 생성 지원 (기존 데이터 위에 추가 생성)
- OpenAI Batch API 지원 (50% 비용 절감)

사용법:
    # 실시간 생성 (기존 방식)
    python scripts/generate_training_data.py

    # Batch API: 요청 파일 생성 + 제출
    python scripts/generate_training_data.py --mode batch

    # Batch 상태 확인
    python scripts/generate_training_data.py --mode batch-status --batch-id batch_xxxxx

    # Batch 결과 다운로드 및 처리
    python scripts/generate_training_data.py --mode batch-download --batch-id batch_xxxxx
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_DIR / "configs" / "training_config.yaml"
OUTPUT_DIR = PROJECT_DIR / "data"
BATCH_DIR = OUTPUT_DIR / "batch"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Genre-specific scene templates (used as seed context for GPT)
# ---------------------------------------------------------------------------
GENRE_TEMPLATES = {
    "drama": {
        "description": "한국 드라마 장면",
        "places": ["사무실", "병원", "학교", "카페", "아파트 거실", "법정", "경찰서", "레스토랑", "공원", "지하철"],
        "times": ["아침", "낮", "오후", "저녁", "밤", "새벽"],
        "atmospheres": ["긴장감 넘치는", "따뜻한", "슬픈", "코믹한", "로맨틱한", "미스터리한", "차가운"],
        "character_types": ["회사원", "의사", "변호사", "형사", "학생", "기자", "사업가", "교사"],
    },
    "movie": {
        "description": "한국 영화 장면",
        "places": ["옥상", "지하 주차장", "항구", "산속 오두막", "고급 호텔", "폐공장", "전통 시장", "해변"],
        "times": ["새벽", "낮", "해질녘", "한밤중"],
        "atmospheres": ["스릴 넘치는", "비장한", "유머러스한", "잔잔한", "공포스러운", "액션감 넘치는"],
        "character_types": ["조직 보스", "경호원", "파일럿", "군인", "탐정", "과학자", "예술가"],
    },
    "documentary": {
        "description": "다큐멘터리 장면",
        "places": ["자연 숲", "북극", "사막", "깊은 바다", "화산 지대", "열대 우림", "도시 거리", "농장"],
        "times": ["이른 아침", "한낮", "일몰", "야간"],
        "atmospheres": ["경이로운", "고요한", "긴박한", "교육적인", "감동적인"],
        "character_types": ["내레이터", "현지 가이드", "연구원", "생물학자", "원주민", "환경 운동가"],
    },
    "news": {
        "description": "뉴스 보도 장면",
        "places": ["뉴스 스튜디오", "국회", "사건 현장", "기자회견장", "거리", "재판소", "공항"],
        "times": ["아침", "낮", "저녁", "긴급 속보"],
        "atmospheres": ["긴급한", "공식적인", "차분한", "충격적인", "논쟁적인"],
        "character_types": ["앵커", "기자", "정치인", "전문가 패널", "시민 인터뷰이", "경찰 대변인"],
    },
    "variety_show": {
        "description": "예능 프로그램 장면",
        "places": ["스튜디오 세트", "야외 촬영지", "게임 무대", "캠핑장", "시골 마을", "놀이공원", "식당"],
        "times": ["낮", "저녁", "밤"],
        "atmospheres": ["유쾌한", "웃긴", "감동적인", "경쟁적인", "훈훈한", "혼란스러운"],
        "character_types": ["MC", "게스트 연예인", "개그맨", "가수", "배우", "일반인 참가자"],
    },
    "animation": {
        "description": "애니메이션 장면",
        "places": ["마법의 숲", "우주선", "학교 교실", "성", "수중 도시", "미래 도시", "구름 위 마을"],
        "times": ["아침", "한낮", "노을", "달밤"],
        "atmospheres": ["판타지적인", "모험적인", "귀여운", "어두운", "희망찬", "슬픈"],
        "character_types": ["주인공 소년/소녀", "마법사", "로봇", "요정", "악당", "동물 캐릭터", "멘토"],
    },
    "music_video": {
        "description": "뮤직비디오 장면",
        "places": ["무대", "도심 거리", "사막", "네온 클럽", "옥상", "해변", "폐건물"],
        "times": ["낮", "밤", "새벽", "황혼"],
        "atmospheres": ["에너지 넘치는", "몽환적인", "감성적인", "강렬한", "자유로운"],
        "character_types": ["아이돌 가수", "밴드 멤버", "댄서", "엑스트라"],
    },
    "advertisement": {
        "description": "광고 장면",
        "places": ["깨끗한 주방", "화려한 거리", "체육관", "자연 배경", "현대적 사무실", "가정집"],
        "times": ["밝은 낮", "따뜻한 오후", "세련된 저녁"],
        "atmospheres": ["밝고 긍정적인", "고급스러운", "친근한", "활동적인", "감각적인"],
        "character_types": ["모델", "가족", "운동선수", "셰프", "직장인", "어린이"],
    },
    "sports": {
        "description": "스포츠 중계 장면",
        "places": ["축구장", "야구장", "수영장", "체육관", "스키장", "육상 트랙", "격투기장"],
        "times": ["낮 경기", "야간 경기", "연장전"],
        "atmospheres": ["열정적인", "긴장감 넘치는", "환호하는", "실망스러운", "극적인"],
        "character_types": ["선수", "감독", "해설자", "심판", "관중", "코치"],
    },
    "education": {
        "description": "교육/강의 장면",
        "places": ["강의실", "실험실", "도서관", "온라인 스튜디오", "야외 현장", "세미나실"],
        "times": ["오전", "오후"],
        "atmospheres": ["학구적인", "호기심 자극하는", "진지한", "편안한", "인터랙티브한"],
        "character_types": ["교수", "강사", "학생", "조교", "전문가 초빙 강연자"],
    },
}


def build_generation_prompt(genre: str, template: dict, batch_size: int) -> str:
    """GPT에게 보낼 프롬프트를 구성합니다."""
    return f"""당신은 미디어 콘텐츠 메타데이터와 검색 질의 데이터셋 생성 전문가입니다.

**장르**: {template['description']}

아래 조건에 맞는 JSON 배열을 {batch_size}개 생성하세요. 각 항목은 하나의 장면(scene)을 나타냅니다.

## 장면 메타데이터 생성 규칙
- Place: {json.dumps(template['places'], ensure_ascii=False)} 중 선택하거나 유사한 장소
- Approximate Time: {json.dumps(template['times'], ensure_ascii=False)} 중 선택
- Atmosphere: {json.dumps(template['atmospheres'], ensure_ascii=False)} 중 선택하거나 유사한 분위기
- Keywords: 장면과 관련된 10~15개의 구체적 키워드 (시각적 요소, 소품, 배경, 소리, 감정 등)
- Main Characters: 2~3명, 각각 type/name/description 포함. 외모, 복장, 표정을 구체적으로 묘사
- caption: 장면의 상세한 서술 (**반드시 3~5문장**). 다음을 모두 포함해야 합니다:
  - 배경 및 공간의 시각적 묘사
  - 각 등장인물이 **구체적으로 무엇을 하고 있는지** 개별적으로 서술
  - 인물 간의 상호작용이나 위치 관계
  - 분위기나 감정적 톤
  - 예시: "넓은 사무실 한쪽에서 영희는 노트북을 펼쳐놓고 분기 보고서를 작성하고 있다. 그 옆 자리에 앉은 민수는 커피잔을 들고 창밖을 바라보며 잠시 쉬고 있다. 팀장 박준혁은 회의실 유리문 너머로 전화 통화를 하며 심각한 표정을 짓고 있다. 사무실 전체에 긴장감이 감돌고 있으며, 곧 있을 실적 발표를 앞두고 모두가 각자의 방식으로 불안감을 다루는 모습이다."
- Action: 6~10개의 구체적 동작/대사 서술. 각 인물의 개별 행동을 명시

## Confusable Scene (혼동 유발 장면) 생성 규칙
각 장면마다 **confusable_scenes** 2개를 반드시 생성하세요.
이것은 원본 장면과 의도적으로 혼동을 유발하도록 설계된 변형 장면입니다.

### 혼동 유발 전략 (아래 중 하나 이상 사용):
1. **Entity-Attribute Swap (인물-행동 교환)**: 같은 인물들이 등장하지만 행동이 서로 뒤바뀜
   - 원본: "영희가 노트북으로 일하고 민수가 커피를 마신다" → 변형: "민수가 노트북으로 일하고 영희가 커피를 마신다"
2. **Action Misattribution (행동 오귀속)**: 특정 인물의 행동을 다른 인물에게 부여
   - 원본: "박 팀장이 전화 중" → 변형: "영희가 전화 중이고 박 팀장은 자리에 앉아있다"
3. **Partial Truth Distortion (부분적 사실 왜곡)**: 장면의 대부분은 동일하나 핵심 세부사항이 다름
   - 원본: "카페에서 계약서에 서명" → 변형: "카페에서 계약서를 검토만 하고 서명하지 않음"
4. **Temporal/Causal Shift (시간/인과 변경)**: 사건의 순서나 원인이 다름
   - 원본: "화가 나서 문을 닫고 나감" → 변형: "문을 닫고 나간 뒤 화가 남"

confusable_scenes의 형식은 원본 metadata와 동일합니다 (Place, Approximate Time, Atmosphere, Keywords, Main Characters, caption, Action).
confusable_scenes는 원본과 **같은 장소, 유사한 인물**을 공유하되 **행동, 대사, 인과관계가 미묘하게 다른** 장면이어야 합니다.

## 질의(Query) 생성 규칙
각 장면에 대해 다음 3종류의 질의를 생성하세요:

### 1. normal (3개): 정답 질의
- 이 장면을 정확히 찾기 위한 자연어 질의
- 사용자가 실제로 검색할 법한 자연스러운 질문
- 예: "영희가 노트북으로 보고서 작성하는 사무실 장면"

### 2. hard_negative (5개): Hard Negative 질의
Dense Embedding 모델이 혼동할 만큼 어려운 질의를 생성하세요.
아래 패턴을 **반드시 섞어서** 사용하세요:

**(a) Entity-Attribute Swap 질의** (2개 이상):
- 장면 내 인물의 행동/속성을 서로 뒤바꾼 질의
- 원본이 "영희가 노트북 작업, 민수가 커피"일 때 → "민수가 노트북으로 일하고 있어?" 또는 "영희가 커피 마시고 있어?"
- 키워드(노트북, 커피, 사무실)가 모두 원본에 존재하므로 Dense Embedding이 높은 유사도를 줄 수 있음

**(b) Action Misattribution 질의** (1개 이상):
- 해당 장면에서 실제로 일어나지 않은 행동을 특정 인물에게 귀속시킨 질의
- "박 팀장이 보고서를 작성하고 있어?" (실제로는 전화 중)

**(c) Subtle Detail 변형 질의** (1개 이상):
- 장면의 핵심 디테일을 미묘하게 바꾼 질의
- "사무실에서 분기보고서 대신 사직서를 작성하는 장면" (문서 종류가 다름)
- "영희가 태블릿으로 작업하는 장면" (기기가 다름)

**(d) Negated/Opposite 질의** (1개 이상):
- 원본 장면의 상황을 부정하거나 반대로 만든 질의
- "사무실에서 모두 여유로운 분위기로 담소하는 장면" (실제로는 긴장감)

### 3. negative (2개): Negative 질의
- 이 장면과 완전히 관련 없는 질의
- 장르, 장소, 인물, 상황 모두 다른 질의

## 출력 형식 (JSON 배열)
```json
[
  {{
    "metadata": {{
      "Place": "...",
      "Approximate Time": "...",
      "Atmosphere": "...",
      "Keywords": ["...", "..."],
      "Main Characters": [
        {{"type": "...", "name": "...", "description": "..."}}
      ],
      "caption": "3~5문장의 상세한 장면 서술...",
      "Action": ["...", "...", "...", "...", "...", "..."]
    }},
    "confusable_scenes": [
      {{
        "Place": "...",
        "Approximate Time": "...",
        "Atmosphere": "...",
        "Keywords": ["...", "..."],
        "Main Characters": [{{"type": "...", "name": "...", "description": "..."}}],
        "caption": "3~5문장의 상세한 변형 장면 서술...",
        "Action": ["...", "...", "...", "...", "...", "..."]
      }},
      {{
        "Place": "...",
        "Approximate Time": "...",
        "Atmosphere": "...",
        "Keywords": ["...", "..."],
        "Main Characters": [{{"type": "...", "name": "...", "description": "..."}}],
        "caption": "3~5문장의 상세한 변형 장면 서술...",
        "Action": ["...", "...", "...", "...", "...", "..."]
      }}
    ],
    "query": {{
      "normal": ["...", "...", "..."],
      "hard_negative": ["...", "...", "...", "...", "..."],
      "negative": ["...", "..."]
    }}
  }}
]
```

**중요 규칙:**
- 모든 텍스트는 한국어로 작성
- caption은 반드시 3~5문장으로 작성. 각 인물의 행동을 개별적으로 서술할 것
- confusable_scenes는 원본과 키워드/장소/인물이 겹치지만 행동/상황이 다른 장면
- hard_negative 질의는 원본 장면의 키워드를 많이 포함하되 의미적으로 원본과 매칭되면 안 됨
- 각 장면은 서로 다른 상황과 인물을 가져야 함
- JSON만 출력하세요. 다른 설명 없이 순수 JSON 배열만 반환하세요.
"""


def call_gpt_api(client: OpenAI, prompt: str, model: str, max_retries: int = 3, retry_delay: float = 2.0) -> list:
    """GPT API를 호출하여 학습 데이터를 생성합니다."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a media content metadata and search query dataset generator. Always respond with valid JSON arrays only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)

            # Handle both {"data": [...]} and [...] formats
            if isinstance(parsed, dict):
                for key in ("data", "scenes", "items", "results"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                if "metadata" in parsed and "query" in parsed:
                    return [parsed]
                return list(parsed.values())[0] if parsed else []
            if isinstance(parsed, list):
                return parsed
            return []

        except json.JSONDecodeError:
            try:
                start = content.index("[")
                end = content.rindex("]") + 1
                return json.loads(content[start:end])
            except (ValueError, json.JSONDecodeError):
                pass
        except Exception as e:
            print(f"  API call attempt {attempt + 1}/{max_retries} failed: {e}")

        if attempt < max_retries - 1:
            wait = retry_delay * (2 ** attempt)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    return []


def validate_scene(scene: dict) -> bool:
    """생성된 장면 데이터의 유효성을 검증합니다."""
    if not isinstance(scene, dict):
        return False

    metadata = scene.get("metadata")
    query = scene.get("query")

    if not isinstance(metadata, dict) or not isinstance(query, dict):
        return False

    required_meta_keys = {"Place", "Approximate Time", "Atmosphere", "Keywords", "Main Characters", "caption", "Action"}
    if not required_meta_keys.issubset(metadata.keys()):
        return False

    if not isinstance(metadata["Keywords"], list) or len(metadata["Keywords"]) < 3:
        return False
    if not isinstance(metadata["Main Characters"], list) or len(metadata["Main Characters"]) < 1:
        return False
    if not isinstance(metadata["Action"], list) or len(metadata["Action"]) < 2:
        return False
    if not isinstance(metadata.get("caption", ""), str) or len(metadata["caption"]) < 30:
        return False

    # Validate confusable_scenes
    confusable = scene.get("confusable_scenes", [])
    if not isinstance(confusable, list) or len(confusable) < 1:
        return False
    for cs in confusable:
        if not isinstance(cs, dict):
            return False
        if not {"Place", "caption", "Action"}.issubset(cs.keys()):
            return False
        if not isinstance(cs.get("caption", ""), str) or len(cs["caption"]) < 20:
            return False

    required_query_keys = {"normal", "hard_negative", "negative"}
    if not required_query_keys.issubset(query.keys()):
        return False

    if not isinstance(query["normal"], list) or len(query["normal"]) < 1:
        return False
    if not isinstance(query["hard_negative"], list) or len(query["hard_negative"]) < 1:
        return False
    if not isinstance(query["negative"], list) or len(query["negative"]) < 1:
        return False

    return True


def load_existing_data() -> tuple[list, dict[str, int]]:
    """기존에 생성된 데이터를 로드하고 장르별 개수를 파악합니다."""
    existing_scenes = []
    genre_counts = {}

    raw_all_path = OUTPUT_DIR / "raw_all.json"
    if raw_all_path.exists():
        with open(raw_all_path, "r", encoding="utf-8") as f:
            existing_scenes = json.load(f)
        for scene in existing_scenes:
            genre = scene.get("genre", "unknown")
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        print(f"기존 데이터 로드 완료: {len(existing_scenes)}개 장면")
        for g, c in sorted(genre_counts.items()):
            print(f"  {g}: {c}개")
    else:
        # Try loading from individual genre files
        for genre_file in OUTPUT_DIR.glob("raw_*.json"):
            if genre_file.name == "raw_all.json":
                continue
            genre_name = genre_file.stem.replace("raw_", "")
            with open(genre_file, "r", encoding="utf-8") as f:
                genre_data = json.load(f)
            existing_scenes.extend(genre_data)
            genre_counts[genre_name] = len(genre_data)
        if existing_scenes:
            print(f"장르별 파일에서 기존 데이터 로드: {len(existing_scenes)}개 장면")

    return existing_scenes, genre_counts


def split_dataset(data: list, config: dict) -> dict:
    """데이터셋을 train/val/test로 분할합니다."""
    random.shuffle(data)
    total = len(data)
    train_end = int(total * config["data_generation"]["train_ratio"])
    val_end = train_end + int(total * config["data_generation"]["val_ratio"])

    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:],
    }


def _parse_gpt_response(content: str) -> list:
    """GPT 응답을 파싱하여 장면 리스트를 반환합니다."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        try:
            start = content.index("[")
            end = content.rindex("]") + 1
            return json.loads(content[start:end])
        except (ValueError, json.JSONDecodeError):
            return []

    if isinstance(parsed, dict):
        for key in ("data", "scenes", "items", "results"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        if "metadata" in parsed and "query" in parsed:
            return [parsed]
        values = list(parsed.values())
        return values[0] if values and isinstance(values[0], list) else []
    if isinstance(parsed, list):
        return parsed
    return []


def _get_client() -> OpenAI:
    """OpenAI 클라이언트를 생성합니다."""
    api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("OPEN_AI_API_KEY not set. Please set it in .env file.")
    return OpenAI(api_key=api_key)


def _save_and_split(all_scenes: list, config: dict):
    """전체 데이터를 저장하고 train/val/test로 분할합니다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw_all.json
    raw_path = OUTPUT_DIR / "raw_all.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_scenes, f, ensure_ascii=False, indent=2)
    print(f"  전체 데이터: {len(all_scenes)}개 => {raw_path}")

    # Save per-genre files
    genre_groups = {}
    for scene in all_scenes:
        g = scene.get("genre", "unknown")
        genre_groups.setdefault(g, []).append(scene)
    for g, scenes_list in genre_groups.items():
        genre_path = OUTPUT_DIR / f"raw_{g}.json"
        with open(genre_path, "w", encoding="utf-8") as f:
            json.dump(scenes_list, f, ensure_ascii=False, indent=2)

    # Split
    splits = split_dataset(all_scenes, config)
    for split_name, split_data in splits.items():
        split_path = OUTPUT_DIR / f"{split_name}.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}: {len(split_data)} samples => {split_path}")

    generate_triplets(splits["train"], OUTPUT_DIR / "train_triplets.json")
    generate_triplets(splits["val"], OUTPUT_DIR / "val_triplets.json")


def _calc_genre_needs(config: dict) -> dict[str, int]:
    """장르별 추가 생성 필요량을 계산합니다."""
    gen_config = config["data_generation"]
    total_target = gen_config["total_samples"]
    genres = gen_config["genres"]

    _, genre_counts = load_existing_data()
    target_per_genre = total_target // len(genres)

    needs = {}
    for genre in genres:
        existing = genre_counts.get(genre, 0)
        needed = target_per_genre - existing
        if needed > 0:
            needs[genre] = needed
        else:
            print(f"[{genre}] 이미 {existing}개 존재 (목표: {target_per_genre}). 건너뜀.")
    return needs


# ---------------------------------------------------------------------------
# OpenAI Batch API
# ---------------------------------------------------------------------------

def create_batch_requests(config: dict) -> Path | None:
    """Batch API용 JSONL 요청 파일을 생성합니다."""
    gen_config = config["data_generation"]
    batch_size = gen_config["api_batch_size"]
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    genre_needs = _calc_genre_needs(config)
    if not genre_needs:
        print("추가 생성이 필요하지 않습니다.")
        return None

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = BATCH_DIR / "batch_requests.jsonl"

    request_count = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for genre, needed in genre_needs.items():
            template = GENRE_TEMPLATES[genre]
            num_batches = max(1, needed // batch_size)
            print(f"[{genre}] {num_batches}개 배치 요청 생성 (필요: {needed}개)")

            for batch_idx in range(num_batches):
                prompt = build_generation_prompt(genre, template, batch_size)
                request = {
                    "custom_id": f"{genre}_{batch_idx:05d}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a media content metadata and search query dataset generator. Always respond with valid JSON arrays only."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.9,
                        "max_tokens": 8192,
                        "response_format": {"type": "json_object"},
                    },
                }
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
                request_count += 1

    print(f"\n총 {request_count}개 배치 요청 생성 => {jsonl_path}")
    return jsonl_path


def submit_batch(client: OpenAI, jsonl_path: Path) -> str:
    """Batch 요청 파일을 업로드하고 배치 작업을 제출합니다."""
    print(f"파일 업로드 중: {jsonl_path}")
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  파일 ID: {uploaded.id}")

    print("배치 작업 제출 중...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "media_embedding_training_data"},
    )
    print(f"  배치 ID: {batch.id}")
    print(f"  상태: {batch.status}")

    # Save batch info
    batch_info = {"batch_id": batch.id, "input_file_id": uploaded.id, "status": batch.status}
    info_path = BATCH_DIR / f"batch_info_{batch.id}.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(batch_info, f, indent=2)

    print(f"\n배치가 제출되었습니다!")
    print(f"상태 확인:   python scripts/generate_training_data.py --mode batch-status --batch-id {batch.id}")
    print(f"결과 다운로드: python scripts/generate_training_data.py --mode batch-download --batch-id {batch.id}")
    return batch.id


def check_batch_status(client: OpenAI, batch_id: str):
    """배치 작업의 상태를 확인합니다."""
    batch = client.batches.retrieve(batch_id)
    print(f"배치 ID: {batch.id}")
    print(f"상태: {batch.status}")
    print(f"총 요청: {batch.request_counts.total}")
    print(f"완료: {batch.request_counts.completed}")
    print(f"실패: {batch.request_counts.failed}")

    if batch.output_file_id:
        print(f"결과 파일 ID: {batch.output_file_id}")
    if batch.error_file_id:
        print(f"에러 파일 ID: {batch.error_file_id}")

    if batch.status == "completed":
        print(f"\n배치 완료! 결과를 다운로드하세요:")
        print(f"  python scripts/generate_training_data.py --mode batch-download --batch-id {batch_id}")
    elif batch.status in ("validating", "in_progress", "finalizing"):
        progress = batch.request_counts.completed / max(batch.request_counts.total, 1) * 100
        print(f"\n진행률: {progress:.1f}%")
    elif batch.status == "failed":
        print("\n배치 실패.")
        if batch.errors:
            for error in batch.errors.data:
                print(f"  에러: {error.code} - {error.message}")


def download_batch_results(client: OpenAI, batch_id: str, config: dict):
    """배치 결과를 다운로드하고 학습 데이터로 처리합니다."""
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"배치가 아직 완료되지 않았습니다 (상태: {batch.status})")
        return

    if not batch.output_file_id:
        print("결과 파일이 없습니다.")
        return

    # Download results
    print(f"결과 다운로드 중 (파일 ID: {batch.output_file_id})...")
    result_content = client.files.content(batch.output_file_id)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    result_path = BATCH_DIR / f"batch_results_{batch_id}.jsonl"
    with open(result_path, "wb") as f:
        f.write(result_content.read())
    print(f"  저장: {result_path}")

    # Download errors if any
    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        error_path = BATCH_DIR / f"batch_errors_{batch_id}.jsonl"
        with open(error_path, "wb") as f:
            f.write(error_content.read())
        print(f"  에러 파일: {error_path}")

    # Parse results
    print("\n결과 파싱 중...")
    new_scenes = []
    failed_count = 0

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line.strip())
            custom_id = result.get("custom_id", "")
            genre = custom_id.rsplit("_", 1)[0] if "_" in custom_id else "unknown"

            if result.get("error"):
                failed_count += 1
                continue

            response_body = result.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            if not choices:
                failed_count += 1
                continue

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                failed_count += 1
                continue

            scenes = _parse_gpt_response(content)
            for scene in scenes:
                if validate_scene(scene):
                    scene["genre"] = genre
                    new_scenes.append(scene)

    print(f"  새로 생성된 장면: {len(new_scenes)}개")
    print(f"  실패: {failed_count}개")

    # Merge with existing data
    existing_scenes, _ = load_existing_data()
    all_scenes = existing_scenes + new_scenes
    print(f"  합계: 기존 {len(existing_scenes)} + 신규 {len(new_scenes)} = {len(all_scenes)}개")

    _save_and_split(all_scenes, config)
    print("\n데이터 처리 완료!")


# ---------------------------------------------------------------------------
# Real-time generation (기존 방식)
# ---------------------------------------------------------------------------

def run_realtime(config: dict):
    """실시간 API 호출로 데이터를 생성합니다."""
    gen_config = config["data_generation"]
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    client = _get_client()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_target = gen_config["total_samples"]
    batch_size = gen_config["api_batch_size"]

    existing_scenes, genre_counts = load_existing_data()
    total_existing = len(existing_scenes)

    if total_existing >= total_target:
        print(f"이미 {total_existing}개 장면이 존재합니다 (목표: {total_target}). 추가 생성이 필요하지 않습니다.")
        print("데이터 분할만 다시 수행합니다...")
        _save_and_split(existing_scenes, config)
        print("\n데이터 분할 완료!")
        return

    genre_needs = _calc_genre_needs(config)
    remaining = sum(genre_needs.values())
    print(f"\n목표: {total_target}개 장면, 기존: {total_existing}개, 추가 생성: {remaining}개\n")

    all_scenes = list(existing_scenes)

    for genre, needed in genre_needs.items():
        template = GENRE_TEMPLATES[genre]
        actual_batches = max(1, needed // batch_size)
        print(f"[{genre}] {needed}개 추가 생성...")

        genre_new_scenes = []
        for batch_idx in tqdm(range(actual_batches), desc=f"  {genre}"):
            prompt = build_generation_prompt(genre, template, batch_size)
            scenes = call_gpt_api(
                client, prompt, model,
                max_retries=gen_config["api_max_retries"],
                retry_delay=gen_config["api_retry_delay"],
            )

            for scene in scenes:
                if validate_scene(scene):
                    scene["genre"] = genre
                    genre_new_scenes.append(scene)

            if batch_idx < actual_batches - 1:
                time.sleep(1)

        print(f"  => {len(genre_new_scenes)} valid scenes generated (new)")
        all_scenes.extend(genre_new_scenes)

    print(f"\n총 장면 수: {len(all_scenes)}")
    _save_and_split(all_scenes, config)
    print("\nData generation complete!")


def generate_triplets(scenes: list, output_path: Path):
    """
    학습용 triplet 데이터를 생성합니다.
    각 triplet: (query, positive_passage, hard_negative_passages, negative_passage)

    Hard negative passage는 confusable_scenes의 passage를 사용합니다.
    이를 통해 랜덤 passage가 아닌 의미적으로 유사하지만 다른 passage가 hard negative로 사용됩니다.
    """
    triplets = []

    # Build a pool of all passages for easy negatives
    all_passages = []
    for scene in scenes:
        all_passages.append(metadata_to_passage(scene["metadata"]))

    for scene_idx, scene in enumerate(scenes):
        metadata = scene["metadata"]
        query_data = scene["query"]
        confusable = scene.get("confusable_scenes", [])

        passage = metadata_to_passage(metadata)

        # Build hard negative passages from confusable scenes
        hard_neg_passages = [metadata_to_passage(cs) for cs in confusable]

        # If confusable_scenes are insufficient, supplement with same-genre scenes
        if len(hard_neg_passages) < 2:
            same_genre = [
                metadata_to_passage(s["metadata"])
                for i, s in enumerate(scenes)
                if i != scene_idx and s.get("genre") == scene.get("genre")
            ]
            if same_genre:
                needed = 2 - len(hard_neg_passages)
                hard_neg_passages.extend(random.sample(same_genre, min(needed, len(same_genre))))

        # Build easy negative passages from random other scenes
        other_passages = [p for i, p in enumerate(all_passages) if i != scene_idx]

        for normal_q in query_data["normal"]:
            neg_passages = random.sample(other_passages, min(2, len(other_passages))) if other_passages else []
            triplet = {
                "query": normal_q,
                "positive": passage,
                "hard_negatives": hard_neg_passages,
                "negatives": neg_passages,
            }
            triplets.append(triplet)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

    print(f"  Triplets: {len(triplets)} => {output_path}")


def metadata_to_passage(metadata: dict) -> str:
    """메타데이터를 자연어 passage로 변환합니다."""
    parts = []

    place = metadata.get("Place", "")
    time_info = metadata.get("Approximate Time", "")
    atmosphere = metadata.get("Atmosphere", "")
    if place or time_info or atmosphere:
        setting = f"{place}에서 {time_info}에 벌어지는 {atmosphere} 장면이다." if all([place, time_info, atmosphere]) else ""
        if not setting:
            setting = f"장소: {place}. 시간: {time_info}. 분위기: {atmosphere}."
        parts.append(setting)

    # Caption is now the core of the passage
    caption = metadata.get("caption", "")
    if caption:
        parts.append(caption)

    # Characters
    characters = metadata.get("Main Characters", [])
    char_descs = []
    for char in characters:
        if isinstance(char, dict):
            char_descs.append(f"{char.get('name', '')}({char.get('type', '')}): {char.get('description', '')}")
        elif isinstance(char, str) and char.strip():
            char_descs.append(char)
    if char_descs:
        parts.append("등장인물: " + ". ".join(char_descs) + ".")

    # Actions
    actions = metadata.get("Action", [])
    if actions:
        parts.append(" ".join(actions))

    # Keywords as context
    keywords = metadata.get("Keywords", [])
    if keywords:
        parts.append("키워드: " + ", ".join(keywords))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BGE-M3 학습 데이터 생성")
    parser.add_argument(
        "--mode", type=str, default="realtime",
        choices=["realtime", "batch", "batch-status", "batch-download"],
        help="실행 모드: realtime(실시간), batch(Batch API 제출), batch-status(상태 확인), batch-download(결과 다운로드)",
    )
    parser.add_argument("--batch-id", type=str, default=None, help="Batch ID (batch-status, batch-download에서 필요)")
    args = parser.parse_args()

    config = load_config()

    if args.mode == "realtime":
        run_realtime(config)

    elif args.mode == "batch":
        client = _get_client()
        jsonl_path = create_batch_requests(config)
        if jsonl_path:
            submit_batch(client, jsonl_path)

    elif args.mode == "batch-status":
        if not args.batch_id:
            print("Error: --batch-id 필요. 예: --batch-id batch_xxxxx")
            return
        client = _get_client()
        check_batch_status(client, args.batch_id)

    elif args.mode == "batch-download":
        if not args.batch_id:
            print("Error: --batch-id 필요. 예: --batch-id batch_xxxxx")
            return
        client = _get_client()
        download_batch_results(client, args.batch_id, config)


if __name__ == "__main__":
    main()
