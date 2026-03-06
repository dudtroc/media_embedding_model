#!/bin/bash
# Conda 환경 설정 스크립트
# 사용법: bash setup_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="media_embedding"

# conda 명령어 확인
if ! command -v conda &> /dev/null; then
    echo "conda가 설치되어 있지 않습니다."
    echo "https://docs.conda.io/en/latest/miniconda.html 에서 Miniconda를 설치해주세요."
    exit 1
fi

# 기존 환경이 있으면 업데이트, 없으면 새로 생성
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "기존 '${ENV_NAME}' 환경을 업데이트합니다..."
    conda env update -f "${SCRIPT_DIR}/environment.yml" --prune
else
    echo "'${ENV_NAME}' 환경을 생성합니다..."
    conda env create -f "${SCRIPT_DIR}/environment.yml"
fi

echo ""
echo "환경 설정이 완료되었습니다!"
echo "다음 명령어로 환경을 활성화하세요:"
echo ""
echo "  conda activate ${ENV_NAME}"
echo ""
echo "학습 실행:"
echo "  python scripts/train.py"
echo ""
echo "테스트 실행:"
echo "  python test.py"
