import os
import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Configure UTF-8 for console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import (
    GEMINI_MODEL,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    SUPPORTED_VIDEO_EXTS,
    SUPPORTED_SUBTITLE_EXTS,
    TRANSLATION_SYSTEM_PROMPT,
    TEMP_DIR,
)
from core.pipeline import SubtitlePipeline
from core.transcriber import WhisperTranscriber
from core.cleaner_translator import GeminiSubtitleCleaner
from core.subtitle_parser import SubtitleParser

# =========================================================
# Page Configuration & Custom CSS
# =========================================================
st.set_page_config(
    page_title="Local Video Subtitle Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e9ecef;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Initialize Session State
# =========================================================
if "batch_files" not in st.session_state:
    st.session_state.batch_files = []
if "batch_logs" not in st.session_state:
    st.session_state.batch_logs = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


# =========================================================
# Sidebar: System & Engine Settings
# =========================================================
with st.sidebar:
    st.header("⚙️ 시스템 및 엔진 설정")

    # API Status Check
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key and "your_gemini_api_key" not in api_key:
        st.success(f"🔑 Gemini API 키 연결됨 (`{api_key[:4]}...{api_key[-3:]}`)")
    else:
        st.error("⚠️ `.env` 파일에 GEMINI_API_KEY가 없습니다.")

    st.markdown("---")
    st.subheader("🧠 모델 설정")
    selected_gemini_model = st.selectbox(
        "Gemini 번역 모델",
        ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash"],
        index=0,
        help="Gemini 3.5 Flash / 3.6 Flash가 가장 빠르고 안정적인 자막 번역 품질을 제공합니다.",
    )

    selected_whisper_model = st.selectbox(
        "Whisper STT 모델",
        ["large-v3", "medium", "small", "base"],
        index=0,
        help="RTX 3090(24GB)에서는 최고 성능의 large-v3를 추천합니다.",
    )

    st.info(f"🎮 **하드웨어 가속**: {WHISPER_DEVICE.upper()} ({WHISPER_COMPUTE_TYPE})")

    st.markdown("---")
    st.subheader("📝 맞춤 번역 지시사항 (선택)")
    custom_instructions = st.text_area(
        "추가 프롬프트",
        placeholder="예: '자연스러운 존댓말 유지, 의학/법률 전문 용어 원어 병기, 등장인물 호칭 일관성 유지'",
        height=100,
    )


# =========================================================
# Header
# =========================================================
st.markdown("<div class='main-title'>🎬 로컬 동영상 AI 한국어 자막 생성기</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>로컬 RTX 3090 GPU (Whisper large-v3) 음성 인식과 Gemini 3.7 Flash 문맥 번역 및 맞춤법/띄어쓰기 정제 엔진</div>",
    unsafe_allow_html=True,
)


# =========================================================
# Main Tabs
# =========================================================
tab_batch, tab_single, tab_settings = st.tabs([
    "📂 폴더 일괄 변환 (Batch)",
    "🎬 단일 파일 변환 & 뷰어 (Single)",
    "🛠️ 고급 설정 & 시스템 정보",
])


# ---------------------------------------------------------
# TAB 1: 폴더 일괄 변환 (Batch Processing)
# ---------------------------------------------------------
with tab_batch:
    st.subheader("📁 폴더 내 동영상 일괄 자막 생성")
    st.markdown("지정한 폴더 안의 모든 동영상을 자동으로 검색하여 순차적으로 고품질 한국어 자막(`.ko.srt`)을 생성합니다.")

    col1, col2 = st.columns([4, 1])
    with col1:
        folder_input = st.text_input(
            "동영상 폴더 경로 입력 (절대 경로)",
            placeholder="예: D:\\Videos\\Lectures 또는 C:\\Users\\user\\Videos",
        )
    with col2:
        st.write("")
        st.write("")
        scan_btn = st.button("🔍 폴더 스캔", use_container_width=True)

    recursive_check = st.checkbox("하위 폴더까지 재귀적으로 검색", value=False)

    if scan_btn and folder_input:
        folder_path = Path(folder_input).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            st.error(f"❌ 올바른 폴더 경로가 아닙니다: `{folder_path}`")
        else:
            files_found = []
            pattern = "**/*" if recursive_check else "*"
            for p in folder_path.glob(pattern):
                if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS:
                    # Check for existing subtitle
                    has_sub = SubtitlePipeline._find_existing_subtitle(p) is not None
                    has_ko = p.with_name(f"{p.stem}.ko.srt").exists()
                    files_found.append({
                        "파일명": p.name,
                        "상대 경로": str(p.relative_to(folder_path)),
                        "용량(MB)": f"{p.stat().st_size / (1024*1024):.1f}",
                        "기존 자막": "✅ 있음" if has_sub else "❌ 없음 (STT 수행)",
                        "한국어 자막(.ko)": "🎉 이미 존재" if has_ko else "대기 중",
                        "full_path": str(p),
                    })

            st.session_state.batch_files = files_found
            if files_found:
                st.success(f"총 {len(files_found)}개의 동영상 파일을 발견했습니다.")
            else:
                st.warning("폴더 내에 지원되는 동영상 파일(.mp4, .mkv, .avi 등)이 없습니다.")

    # Show scanned files table
    if st.session_state.batch_files:
        df = pd.DataFrame(st.session_state.batch_files)
        display_df = df.drop(columns=["full_path"])
        st.dataframe(display_df, use_container_width=True)

        col_start, col_clear = st.columns([2, 1])
        with col_start:
            start_batch_btn = st.button(
                "🚀 전체 일괄 변환 시작",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.is_processing,
            )
        with col_clear:
            if st.button("목록 초기화", use_container_width=True):
                st.session_state.batch_files = []
                st.rerun()

        if start_batch_btn:
            st.session_state.is_processing = True
            total_items = len(st.session_state.batch_files)

            overall_prog = st.progress(0, text="일괄 처리 준비 중...")
            current_item_status = st.empty()
            log_container = st.container()

            # Initialize Pipeline with selected models
            custom_stt = WhisperTranscriber(model_size=selected_whisper_model)
            custom_cleaner = GeminiSubtitleCleaner(model_name=selected_gemini_model)
            batch_pipeline = SubtitlePipeline(stt=custom_stt, cleaner=custom_cleaner)

            success_count = 0
            for idx, item in enumerate(st.session_state.batch_files, start=1):
                target_path = Path(item["full_path"])
                current_item_status.info(f"🎬 **[{idx}/{total_items}] 처리 중**: `{target_path.name}`")

                def update_progress(msg: str, pct: float):
                    current_item_status.markdown(f"🎬 **[{idx}/{total_items}] `{target_path.name}`**\n> ⏳ {msg} ({pct:.0f}%)")

                try:
                    out_srt = batch_pipeline.process_file(
                        input_path=target_path,
                        custom_instructions=custom_instructions,
                        progress_callback=update_progress,
                    )
                    success_count += 1
                    with log_container:
                        st.success(f"✅ [{idx}/{total_items}] 완료: `{out_srt.name}`")
                except Exception as ex:
                    with log_container:
                        st.error(f"❌ [{idx}/{total_items}] 실패 `{target_path.name}`: {ex}")

                overall_prog.progress(idx / total_items, text=f"전체 진행률: {idx}/{total_items} ({idx/total_items*100:.1f}%)")

            st.session_state.is_processing = False
            st.balloons()
            st.success(f"🎉 모든 일괄 작업 완료! (성공: {success_count}/{total_items})")


# ---------------------------------------------------------
# TAB 2: 단일 파일 변환 & 뷰어 (Single Processing & Editor)
# ---------------------------------------------------------
with tab_single:
    st.subheader("🎬 개별 동영상 또는 자막 파일 변환")
    st.markdown("단일 파일의 경로를 입력하거나 업로드하여 변환하고, 생성된 한국어 자막을 바로 확인 및 다운로드할 수 있습니다.")

    single_method = st.radio("파일 지정 방식", ["로컬 파일 경로 직접 입력 (추천, 대용량 영상 지원)", "웹 브라우저 파일 업로드"], horizontal=True)

    target_single_path = None

    if single_method == "로컬 파일 경로 직접 입력 (추천, 대용량 영상 지원)":
        single_path_input = st.text_input(
            "동영상 또는 자막 파일의 전체 경로",
            placeholder="예: D:\\Movies\\lecture.mp4 또는 D:\\Subtitles\\dirty.srt",
        )
        if single_path_input and Path(single_path_input).exists():
            target_single_path = Path(single_path_input)
            st.success(f"파일 확인됨: `{target_single_path.name}` ({target_single_path.stat().st_size / (1024*1024):.2f} MB)")
    else:
        uploaded_file = st.file_uploader(
            "동영상 또는 자막 파일 업로드",
            type=["mp4", "mkv", "avi", "mov", "webm", "srt", "vtt", "ass"],
        )
        if uploaded_file:
            temp_save = TEMP_DIR / uploaded_file.name
            with open(temp_save, "wb") as f:
                f.write(uploaded_file.getbuffer())
            target_single_path = temp_save
            st.success(f"파일 업로드 완료: `{uploaded_file.name}`")

    col_lang, col_btn = st.columns([2, 2])
    with col_lang:
        lang_hint = st.selectbox(
            "음성 언어 힌트",
            ["자동 감지 (Auto)", "영어 (en)", "일본어 (ja)", "중국어 (zh)", "한국어 (ko)"],
            index=0,
        )
        lang_code = None
        if "en" in lang_hint:
            lang_code = "en"
        elif "ja" in lang_hint:
            lang_code = "ja"
        elif "zh" in lang_hint:
            lang_code = "zh"
        elif "ko" in lang_hint:
            lang_code = "ko"

    with col_btn:
        st.write("")
        st.write("")
        run_single_btn = st.button("✨ 한국어 자막 생성 시작", type="primary", use_container_width=True, disabled=target_single_path is None)

    if run_single_btn and target_single_path:
        single_prog = st.progress(0, text="자막 생성 파이프라인 시작...")
        single_status = st.empty()

        def single_callback(msg: str, pct: float):
            single_status.info(f"⏳ {msg}")
            single_prog.progress(int(pct), text=f"{msg} ({pct:.0f}%)")

        custom_stt = WhisperTranscriber(model_size=selected_whisper_model)
        custom_cleaner = GeminiSubtitleCleaner(model_name=selected_gemini_model)
        single_pipeline = SubtitlePipeline(stt=custom_stt, cleaner=custom_cleaner)

        try:
            start_time = time.time()
            out_file = single_pipeline.process_file(
                input_path=target_single_path,
                source_language=lang_code,
                custom_instructions=custom_instructions,
                progress_callback=single_callback,
            )
            elapsed = time.time() - start_time

            single_status.success(f"🎉 자막 생성 완료! (소요 시간: {elapsed:.1f}초)")
            single_prog.progress(100, text="완료!")

            # Load and display generated subtitle
            generated_items = SubtitleParser.load(out_file)
            sub_table_data = []
            for item in generated_items:
                sub_table_data.append({
                    "No": item.index,
                    "시작 시간": item.start_str,
                    "종료 시간": item.end_str,
                    "한국어 자막": item.text,
                })

            st.markdown("---")
            st.subheader(f"📄 생성된 자막 미리보기 (`{out_file.name}`)")

            # Download Button
            with open(out_file, "r", encoding="utf-8") as f:
                srt_content = f.read()

            st.download_button(
                label="💾 .ko.srt 자막 파일 다운로드",
                data=srt_content,
                file_name=out_file.name,
                mime="text/plain",
                type="primary",
            )

            st.dataframe(pd.DataFrame(sub_table_data), use_container_width=True, height=400)

        except Exception as e:
            single_status.error(f"❌ 변환 중 오류 발생: {e}")


# ---------------------------------------------------------
# TAB 3: 세부 설정 & 시스템 정보
# ---------------------------------------------------------
with tab_settings:
    st.subheader("🛠️ 시스템 환경 및 기본 프롬프트 설정")

    st.markdown("#### 📌 현재 기본 자막 번역 시스템 프롬프트")
    st.code(TRANSLATION_SYSTEM_PROMPT, language="markdown")

    st.markdown("---")
    st.markdown("#### 💻 하드웨어 및 라이브러리 정보")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write(f"- **Python 버전**: `{sys.version.split()[0]}`")
        st.write(f"- **기본 STT 모델**: `faster-whisper {selected_whisper_model}`")
    with col_s2:
        st.write(f"- **가속 디바이스**: `CUDA (RTX 3090 24GB)`")
        st.write(f"- **기본 번역 모델**: `{selected_gemini_model}`")
