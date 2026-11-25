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
def get_hardware_config():
    """
    Automatically detects available hardware (NVIDIA GPU vs CPU)
    and selects the optimal device and compute type.
    """
    env_device = os.getenv("WHISPER_DEVICE")
    env_compute = os.getenv("WHISPER_COMPUTE_TYPE")
    if env_device and env_compute:
        return env_device, env_compute, f"{env_device.upper()} ({env_compute})"

    try:
        import torch
        import ctranslate2
        if torch.cuda.is_available() and ctranslate2.get_cuda_device_count() > 0:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            supported = ctranslate2.get_supported_compute_types("cuda")
            if "float16" in supported and vram_gb >= 6.0:
                return "cuda", "float16", f"NVIDIA GPU ({gpu_name}, VRAM {vram_gb:.1f}GB)"
            elif "int8_float16" in supported:
                return "cuda", "int8_float16", f"NVIDIA GPU ({gpu_name}, VRAM {vram_gb:.1f}GB - 경량 가속)"
            else:
                return "cuda", "float32", f"NVIDIA GPU ({gpu_name}, VRAM {vram_gb:.1f}GB)"
    except Exception:
        pass

    return "cpu", "int8", "CPU 모드 (인텔/AMD 프로세서 INT8 가속)"

WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_HARDWARE_DESC = get_hardware_config()
# Model options: 'large-v3', 'large-v2', 'medium', 'small', 'base'
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
# Voice Activity Detection (VAD) filter:
# False (권장): 옛날 전화기/무전기 필터 음성, BGM/음향 효과가 깔린 대사, 속삭임 등의 누락을 완벽히 방지합니다.
# Whisper 내장 hallucination_silence_threshold(2.0s) 및 no_speech_threshold가 환각과 무음을 안전하게 제어합니다.
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "False").lower() in ("true", "1", "yes")

# ==========================================
# 3. Gemini LLM Translation & Cleaning Settings
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Default model for subtitle translation & grammar correction
# gemini-3.1-flash-lite provides high speed, high stability and complete 100% chunk translation without early stops
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Max subtitle lines per translation chunk (30 lines guarantees 100% 1:1 ID alignment with zero drift)
SUBTITLE_CHUNK_SIZE = 30

# System prompt for high-quality subtitle translation and spacing/grammar correction
TRANSLATION_SYSTEM_PROMPT = """당신은 넷플릭스 및 OTT 전문 영상 자막 번역가이자 한국어 교정 전문가입니다.
주어진 자막 JSON 배열(`[{"id": 번호, "text": "원문"}]`)을 읽고, 다음 규칙을 철저히 준수하여 최고 품질의 자연스러운 한국어 자막 JSON 배열로 변환하십시오.

[필수 번역 및 교정 규칙]
1. 인덱스 1:1 매칭 및 병합 절대 금지 (최우선 규칙):
   - 입력된 모든 `id`에 대해 누락, 추가, 병합 없이 **정확히 1:1로 대응**하는 항목을 출력하십시오.
   - 절대로 인접한 두 자막 항목을 하나로 합치거나(병합 금지), 한 항목의 대사를 앞뒤 번호(`id`)로 이동시키지 마십시오.
   - 입력된 `id` 목록과 출력된 `id` 목록은 개수와 번호가 100% 동일해야 합니다.
2. 자연스러운 구어체 및 문맥 반영: 앞뒤 문맥과 화자의 감정/뉘앙스를 파악하여 직역투를 배제하고 자연스러운 한국어 구어체 자막으로 의역하십시오.
3. 맞춤법 및 띄어쓰기 철저 교정: 한국어 맞춤법 규정과 표준 띄어쓰기를 완벽하게 준수하십시오.
4. 다중 화자 대화 포맷팅 (하이픈 및 줄바꿈 분리):
   - 한 자막 항목(`id`) 내에 질문과 답변처럼 2명 이상의 대화가 섞여 있거나 슬래시(/)로 구분된 경우, 슬래시를 제거하고 넷플릭스 표준 대화 형식인 각 행 앞 하이픈(`- `)과 줄바꿈(`\\n`)으로 명확히 두 줄로 분리하십시오.
   - 예시 1: `갈 도닉이야. / 그 갈 도닉?` ➔ `- 갈 도닉이야.\\n- 그 갈 도닉?`
   - 예시 2: `어떤 일에 압박을 느끼시나요? 뭐랄까, 무리한 난제를` ➔ `- 어떤 일에 압박을 느끼시나요?\\n- 뭐랄까, 무리한 난제를`
5. 화자 태그 배제:
   - `[화자 1]`, `[화자 2]`, `[Speaker 1]`, `[Speaker 2]` 같은 기계적인 화자 라벨은 시청 몰입을 방해하므로 절대 출력하지 마십시오.
6. 문맥 파악 (줄 이동 엄격 금지):
   - 앞뒤 자막의 문맥을 참고하여 자연스럽게 번역하되, 각 `id`에는 반드시 해당 `id`의 원문에 해당하는 번역만 담으십시오. 다음 번호의 대사를 미리 당겨오거나 앞 번호로 넘기지 마십시오.
7. 고유명사 및 전문용어 일관성: 인명, 지명, 작품 내 설정 용어 등은 표준 한국어 표기법 또는 널리 쓰이는 표기법으로 일관되게 번역하십시오.
8. 출력 형식: 반드시 입력과 동일한 `id`를 가진 JSON 배열 `[{"id": 번호, "text": "번역된 한국어 자막"}]` 형식으로만 출력하십시오.
"""

# ==========================================
# 4. Supported File Formats
# ==========================================
SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".wmv", ".ts", ".m4v"}
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
SUPPORTED_SUBTITLE_EXTS = {".srt", ".vtt", ".ass", ".ssa", ".sub", ".smi", ".sami"}
