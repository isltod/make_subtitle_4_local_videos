# 🎬 로컬 동영상 AI 한국어 자막 생성기 (Local Video Subtitle Generator)

로컬 저장소의 동영상 파일 또는 기존 외국어/오탈자 자막을 **NVIDIA GPU (RTX 3090) Whisper 음성 인식**과 **Google Gemini Flash AI 번역/교정**을 통해 고품질의 자연스러운 한국어 자막(`.ko.srt`)으로 변환하는 올인원 도구입니다.

---

## 🌟 주요 특징

1. **하이브리드 AI 파이프라인**:
   - **음성 인식 (STT)**: 로컬 RTX 3090 (24GB) GPU를 활용한 `faster-whisper large-v3` (초고속 및 Silero VAD 무음 필터링)
   - **번역 및 텍스트 교정**: `Google Gemini Flash (3.5 / 3.6 / 3.7)` 기반 문맥 의역 + 한국어 맞춤법/띄어쓰기 정제
2. **다양한 작업 모드**:
   - **자막 없는 동영상**: 오디오 추출 ➔ 음성인식(STT) ➔ Gemini 번역/교정 ➔ `.ko.srt` 생성
   - **기존 외국어/오탈자 자막**: 기존 자막 파싱 ➔ Gemini 번역/교정 ➔ `.ko.srt` 생성
3. **폴더 일괄(Batch) 처리 & 단일 정밀 처리**:
   - 수십 개의 동영상이 있는 폴더를 통째로 스캔하여 원클릭 일괄 변환
4. **시각적 웹 대시보드 (Streamlit)** & **터미널 CLI** 동시 지원

---

## 🛠️ 사전 준비 (Setup)

1. **환경 변수 설정**:
   - `.env.example` 파일을 복사하여 `.env` 파일을 생성하고 [Google AI Studio](https://aistudio.google.com/)에서 발급받은 API 키를 입력합니다.
   ```env
   GEMINI_API_KEY="your_gemini_api_key_here"
   ```
2. **필수 패키지 설치**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 빠른 실행 방법

### 1. 웹 GUI 대시보드 실행 (추천)
- 프로젝트 폴더에서 **`run_app.bat`** 파일을 더블클릭하거나, 터미널에서 다음 명령어를 실행합니다:
```bash
.\.venv\Scripts\streamlit.exe run app.py
```
- 브라우저(`http://localhost:8501`)가 자동으로 열립니다.

### 2. 터미널(CLI) 명령어로 실행

#### ① 단일 동영상 또는 자막 파일 변환
```bash
.\.venv\Scripts\python.exe main.py "D:\Videos\lecture.mp4"
```

#### ② 폴더 내 모든 동영상 일괄(Batch) 변환
```bash
.\.venv\Scripts\python.exe main.py --folder "D:\Videos\Movies"
```

#### ③ 맞춤 번역 프롬프트 추가 전달
```bash
.\.venv\Scripts\python.exe main.py "D:\Videos\tech.mp4" --instructions "격식 있는 존댓말 유지, IT 전문 용어 원어 병기"
```

---

## 📂 프로젝트 구조

```text
make_subtitle_4_local_videos/
├── .venv/                      # Python 가상환경
├── .env                        # Gemini API 키 설정 파일
├── config.py                   # 모델 및 프롬프트 기본 설정
├── app.py                      # Streamlit 웹 GUI 대시보드
├── main.py                     # CLI 터미널 실행기
├── run_app.bat                 # 윈도우 원클릭 실행 배치 파일
│
├── core/                       # 핵심 처리 엔진
│   ├── audio_extractor.py      # FFmpeg 16kHz WAV 오디오 추출기
│   ├── transcriber.py          # faster-whisper STT 엔진 (GPU 가속)
│   ├── cleaner_translator.py   # Gemini 3.7 Flash 번역 및 맞춤법/띄어쓰기 정제기
│   ├── subtitle_parser.py      # 자막 읽기/쓰기/타임코드 파서
│   └── pipeline.py             # 전체 작업 흐름 총괄 파이프라인
│
└── tests/                      # 테스트용 샘플 데이터
```
