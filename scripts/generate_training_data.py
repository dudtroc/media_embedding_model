"""
BGE-M3 Hard Negative 개선을 위한 학습 데이터 생성 스크립트.

GPT API를 사용하여 다양한 미디어 장르의 장면 메타데이터와
normal / hard_negative / negative 질의를 생성합니다.
"""

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
- Keywords: 장면과 관련된 8~12개의 구체적 키워드 (시각적 요소, 소품, 배경 등)
- Main Characters: 1~3명, 각각 type/name/description 포함. 외모, 복장, 표정을 구체적으로 묘사
- caption: 장면의 전체 요약 (1~2문장)
- Action: 4~8개의 구체적 동작/대사 서술

## 질의(Query) 생성 규칙
각 장면에 대해 다음 3종류의 질의를 생성하세요:

### 1. normal (3개): 정답 질의
- 이 장면을 정확히 찾기 위한 자연어 질의
- 사용자가 실제로 검색할 법한 자연스러운 질문
- 예: "은행에서 지점장이 퇴근을 지시하는 장면", "최한수가 걱정스러운 표정으로 서 있는 은행 사무실"

### 2. hard_negative (3개): Hard Negative 질의
- 키워드나 상황이 유사하지만 **이 장면이 아닌 다른 장면**을 찾는 질의
- 의미적으로 미묘하게 다른 질의 (같은 장소/인물이지만 다른 상황, 또는 유사한 상황이지만 다른 장소)
- 예시 (은행 지점장 장면의 hard_negative):
  - "은행 지점장이 고객에게 대출을 설명하는 장면" (같은 인물, 다른 행동)
  - "사무실에서 직원들이 회의하는 장면" (같은 장소, 다른 상황)
  - "직장 상사가 밝은 표정으로 직원을 격려하는 장면" (유사 역할, 다른 감정)

### 3. negative (2개): Negative 질의
- 이 장면과 완전히 관련 없는 질의
- 장르, 장소, 인물, 상황 모두 다른 질의
- 예: "바닷가에서 서핑하는 청년", "산꼭대기에서 일출을 보는 등산객"

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
      "caption": "...",
      "Action": ["...", "..."]
    }},
    "query": {{
      "normal": ["...", "...", "..."],
      "hard_negative": ["...", "...", "..."],
      "negative": ["...", "..."]
    }}
  }}
]
```

**중요 규칙:**
- 모든 텍스트는 한국어로 작성
- 각 장면은 서로 다른 상황과 인물을 가져야 함
- hard_negative 질의는 임베딩 모델이 혼동할 만큼 메타데이터와 유사하되 의미적으로 달라야 함
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
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)

            # Handle both {"data": [...]} and [...] formats
            if isinstance(parsed, dict):
                for key in ("data", "scenes", "items", "results"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                # If dict has numeric keys or is a single scene
                if "metadata" in parsed and "query" in parsed:
                    return [parsed]
                return list(parsed.values())[0] if parsed else []
            if isinstance(parsed, list):
                return parsed
            return []

        except json.JSONDecodeError:
            # Try to extract JSON array from response
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

    if not metadata or not query:
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


def main():
    config = load_config()
    gen_config = config["data_generation"]

    api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("Error: OPEN_AI_API_KEY not set. Please set it in .env file.")
        return

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(api_key=api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_target = gen_config["total_samples"]
    batch_size = gen_config["api_batch_size"]
    genres = gen_config["genres"]

    # Distribute samples across genres evenly
    samples_per_genre = total_target // len(genres)
    batches_per_genre = max(1, samples_per_genre // batch_size)

    all_scenes = []
    print(f"Target: {total_target} samples across {len(genres)} genres")
    print(f"  ~{samples_per_genre} samples/genre, {batches_per_genre} batches/genre (batch_size={batch_size})")
    print()

    for genre in genres:
        template = GENRE_TEMPLATES[genre]
        print(f"[{genre}] Generating {samples_per_genre} samples...")

        genre_scenes = []
        for batch_idx in tqdm(range(batches_per_genre), desc=f"  {genre}"):
            prompt = build_generation_prompt(genre, template, batch_size)
            scenes = call_gpt_api(
                client, prompt, model,
                max_retries=gen_config["api_max_retries"],
                retry_delay=gen_config["api_retry_delay"],
            )

            valid_scenes = []
            for scene in scenes:
                if validate_scene(scene):
                    scene["genre"] = genre
                    valid_scenes.append(scene)

            genre_scenes.extend(valid_scenes)

            # Rate limiting
            if batch_idx < batches_per_genre - 1:
                time.sleep(1)

        print(f"  => {len(genre_scenes)} valid scenes generated")
        all_scenes.extend(genre_scenes)

        # Save each scene as an individual file
        genre_dir = OUTPUT_DIR / "scenes" / genre
        genre_dir.mkdir(parents=True, exist_ok=True)
        for scene_idx, scene in enumerate(genre_scenes):
            scene_path = genre_dir / f"scene_{scene_idx:04d}.json"
            with open(scene_path, "w", encoding="utf-8") as f:
                json.dump(scene, f, ensure_ascii=False, indent=2)

    print(f"\nTotal valid scenes: {len(all_scenes)}")

    # Save complete raw dataset
    raw_path = OUTPUT_DIR / "raw_all.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_scenes, f, ensure_ascii=False, indent=2)

    # Split into train/val/test
    splits = split_dataset(all_scenes, config)
    for split_name, split_data in splits.items():
        # Save combined split file
        split_path = OUTPUT_DIR / f"{split_name}.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)

        # Save each scene as an individual file under split directory
        split_dir = OUTPUT_DIR / "scenes" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for scene_idx, scene in enumerate(split_data):
            scene_path = split_dir / f"scene_{scene_idx:04d}.json"
            with open(scene_path, "w", encoding="utf-8") as f:
                json.dump(scene, f, ensure_ascii=False, indent=2)

        print(f"  {split_name}: {len(split_data)} samples => {split_path} + {split_dir}/")

    # Generate training triplets file (for direct training consumption)
    generate_triplets(splits["train"], OUTPUT_DIR / "train_triplets.json")
    generate_triplets(splits["val"], OUTPUT_DIR / "val_triplets.json")

    print("\nData generation complete!")


def generate_triplets(scenes: list, output_path: Path):
    """
    학습용 triplet 데이터를 생성합니다.
    각 triplet: (query, positive_passage, hard_negative_passage, negative_passage)
    """
    triplets = []

    for scene in scenes:
        metadata = scene["metadata"]
        query_data = scene["query"]

        # Flatten metadata into a passage string
        passage = metadata_to_passage(metadata)

        for normal_q in query_data["normal"]:
            triplet = {
                "query": normal_q,
                "positive": passage,
                "hard_negatives": query_data["hard_negative"],
                "negatives": query_data["negative"],
            }
            triplets.append(triplet)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

    print(f"  Triplets: {len(triplets)} => {output_path}")


def metadata_to_passage(metadata: dict) -> str:
    """메타데이터를 하나의 텍스트 passage로 변환합니다."""
    parts = [
        f"장소: {metadata.get('Place', '')}",
        f"시간: {metadata.get('Approximate Time', '')}",
        f"분위기: {metadata.get('Atmosphere', '')}",
        f"키워드: {', '.join(metadata.get('Keywords', []))}",
    ]

    characters = metadata.get("Main Characters", [])
    for char in characters:
        parts.append(f"등장인물: {char.get('name', '')} ({char.get('type', '')}) - {char.get('description', '')}")

    parts.append(f"요약: {metadata.get('caption', '')}")

    actions = metadata.get("Action", [])
    if actions:
        parts.append(f"행동: {' '.join(actions)}")

    return " | ".join(parts)


if __name__ == "__main__":
    main()
