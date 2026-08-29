import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# ==========================================
# 1. Project Directory Settings
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_DIR = PROJECT_ROOT / ".temp"
TEMP_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. Hardware & STT (Whisper) Settings
# ==========================================
# Model options: 'large-v3', 'large-v2', 'medium', 'small', 'base'
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
# Device options: 'cuda', 'cpu'
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
# Compute types: 'float16', 'bfloat16', 'int8_float16', 'int8'
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
# Enable Voice Activity Detection filter to remove silence/noise
WHISPER_VAD_FILTER = True

# ==========================================
# 3. Gemini LLM Translation & Cleaning Settings
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Default model for subtitle translation & grammar correction
# gemini-3.5-flash provides high speed, high stability and generous free-tier quotas
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Max subtitle lines per translation chunk (100 lines translates in ~4-5s per step)
SUBTITLE_CHUNK_SIZE = 100

# System prompt for high-quality subtitle translation and spacing/grammar correction
TRANSLATION_SYSTEM_PROMPT = """당신은 넷플릭스 및 OTT 전문 영상 자막 번역가이자 한국어 교정 전문가입니다.
주어진 자막 목록(타임코드 인덱스와 텍스트)을 읽고, 다음 규칙을 철저히 준수하여 최고 품질의 자연스러운 한국어 자막으로 변환하십시오.

[필수 번역 및 교정 규칙]
1. 인덱스 1:1 매칭 보존: 입력된 모든 번호(ID)에 대해 누락이나 병합 없이 정확히 1:1로 대응하는 한국어 번역문을 출력하십시오.
2. 자연스러운 구어체 및 문맥 반영: 앞뒤 문맥과 화자의 감정/뉘앙스를 파악하여 직역투를 배제하고 자연스러운 한국어 구어체 자막으로 의역하십시오.
3. 맞춤법 및 띄어쓰기 철저 교정: 한국어 맞춤법 규정과 표준 띄어쓰기를 완벽하게 준수하십시오.
4. 다중 화자 대화 포맷팅 (슬래시 제거):
   - 한 자막 내에 2명 이상의 대화가 섞여 있거나 슬래시(/)로 구분된 경우, 슬래시를 제거하고 넷플릭스 표준 대화 형식인 하이픈(`- `)과 줄바꿈으로 깔끔하게 분리하십시오.
   - 예시: `갈 도닉이야. / 그 갈 도닉?` ➔ `- 갈 도닉이야.\\n- 그 갈 도닉?`
5. 화자 태그 배제:
   - `[화자 1]`, `[화자 2]`, `[Speaker 1]`, `[Speaker 2]` 같은 기계적인 화자 라벨은 시청 몰입을 방해하므로 절대 출력하지 마십시오.
6. 문맥 파절 복원:
   - 앞뒤 자막 간에 문장이 조사나 단어 중간에서 어색하게 끊겨 있는 경우, 전체 문맥을 이해하여 자연스럽고 온전한 의미가 통하도록 매끄럽게 다듬으십시오.
7. 고유명사 및 전문용어 일관성: 인명, 지명, 작품 내 설정 용어 등은 표준 한국어 표기법 또는 널리 쓰이는 표기법으로 일관되게 번역하십시오.
8. 출력 형식: 각 줄은 반드시 `ID | 번역된 한국어 자막` 형식으로만 출력하십시오. 설명이나 부연 설명은 절대 추가하지 마십시오.
"""

# ==========================================
# 4. Supported File Formats
# ==========================================
SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".wmv", ".ts", ".m4v"}
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
SUPPORTED_SUBTITLE_EXTS = {".srt", ".vtt", ".ass", ".ssa", ".sub"}
