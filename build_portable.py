import os
import sys
import shutil
import zipfile
import time
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent
DIST_DIR = REPO_ROOT / "dist"
PORTABLE_DIR = DIST_DIR / "make_subtitle_portable"
ZIP_PATH = DIST_DIR / "make_subtitle_portable.zip"

PY_BASE = Path(getattr(sys, "base_prefix", r"C:\Program Files\Python310"))
VENV_LIB = REPO_ROOT / ".venv" / "Lib" / "site-packages"


def log(msg: str):
    print(f"[*] {msg}", flush=True)


def find_ffmpeg() -> tuple[Path, Path]:
    """Finds ffmpeg.exe and ffprobe.exe on system."""
    import shutil as sh
    ffmpeg_cmd = sh.which("ffmpeg")
    ffprobe_cmd = sh.which("ffprobe")
    if not ffmpeg_cmd or not ffprobe_cmd:
        raise FileNotFoundError("ffmpeg.exe 또는 ffprobe.exe를 시스템에서 찾을 수 없습니다.")
    return Path(ffmpeg_cmd).resolve(), Path(ffprobe_cmd).resolve()


def copy_tree_filtered(src: Path, dst: Path, ignore_names=(".pyc", "__pycache__", ".git")):
    """Fast directory copy ignoring caches and pyc files."""
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in ignore_names]
        rel_path = Path(root).relative_to(src)
        target_dir = dst / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            if any(f.endswith(ext) for ext in ignore_names):
                continue
            src_file = Path(root) / f
            dst_file = target_dir / f
            if not dst_file.exists() or src_file.stat().st_mtime > dst_file.stat().st_mtime:
                shutil.copy2(src_file, dst_file)


def create_batch_files(portable_root: Path):
    """Creates convenient one-click launcher batch files."""
    start_bat_content = """@echo off
chcp 65001 > nul
title 로컬 동영상 AI 자막 생성기 실행 중...
cd /d "%~dp0"

set "APP_DIR=%~dp0"
set "PATH=%APP_DIR%bin;%APP_DIR%python;%APP_DIR%python\\Scripts;%PATH%"

echo ===================================================================
echo   🎬 로컬 동영상 AI 자막 생성기 (Portable Edition)
echo ===================================================================
echo.
echo [*] 파이썬 및 AI 엔진을 초기화하고 있습니다...
echo [*] 잠시 후 기본 웹 브라우저가 자동으로 실행됩니다.
echo.
echo [*] 브라우저 창이 열리면 화면에서 자막 생성을 진행하시면 됩니다.
echo [*] 프로그램을 종료하려면 이 콘솔 창을 닫으십시오.
echo ===================================================================
echo.

"%APP_DIR%python\\python.exe" -m streamlit run "%APP_DIR%app.py" --server.headless=false --server.port=8501 --browser.gatherUsageStats=false

if errorlevel 1 (
    echo.
    echo [!] 프로그램 실행 중 문제가 발생했습니다.
    pause
)
"""

    debug_bat_content = """@echo off
chcp 65001 > nul
title [디버그 모드] 로컬 동영상 AI 자막 생성기
cd /d "%~dp0"

set "APP_DIR=%~dp0"
set "PATH=%APP_DIR%bin;%APP_DIR%python;%APP_DIR%python\\Scripts;%PATH%"

echo [디버그 모드 실행] 상세 로그가 콘솔에 표시됩니다.
"%APP_DIR%python\\python.exe" -m streamlit run "%APP_DIR%app.py" --server.headless=false --server.port=8501
pause
"""

    (portable_root / "실행하기.bat").write_text(start_bat_content, encoding="utf-8")
    (portable_root / "실행하기_디버그.bat").write_text(debug_bat_content, encoding="utf-8")
    log("실행 배치 파일(실행하기.bat, 실행하기_디버그.bat) 생성 완료")


def create_readme(portable_root: Path):
    """Creates a user-friendly README guide."""
    readme_content = """===================================================================
🎬 로컬 동영상 AI 자막 생성기 (포터블 무설치 배포판)
===================================================================

파이썬(Python)이나 기타 도구를 전혀 설치할 필요 없이,
더블 클릭 한 번으로 바로 실행할 수 있는 무설치 포터블 버전입니다.

-------------------------------------------------------------------
[1] 실행 방법
-------------------------------------------------------------------
1. 본 폴더 내의 [실행하기.bat] 파일을 더블 클릭합니다.
2. 검은색 콘솔 창이 열리며 약 2~3초 후 기본 웹 브라우저에
   자막 생성기 대시보드 화면이 자동으로 열립니다.
   (주소: http://localhost:8501)
3. 화면 왼쪽 사이드바에서 [Gemini API 키]를 입력하시면 즉시 준비 완료!
   (Google AI Studio에서 무료로 1분 만에 발급받으실 수 있습니다)

-------------------------------------------------------------------
[2] 주요 기능
-------------------------------------------------------------------
- 단일 동영상 / 자막 변환 및 폴더 일괄 변환 (Batch)
- NVIDIA GPU (RTX / GTX) 자동 감지 및 초고속 CUDA 가속
- 최신 Whisper 음성 인식 기반 정밀 타임코드 추출
- 구글 최신 Gemini AI를 통한 넷플릭스 OTT 표준 한국어 번역 & 교정
- 화자 자동 분리 및 어색한 줄바꿈/단어 끊김 원천 방지

-------------------------------------------------------------------
[3] 폴더 구조
-------------------------------------------------------------------
- 실행하기.bat         : 일반 사용자용 원클릭 실행 아이콘
- 실행하기_디버그.bat  : 오류 발생 시 콘솔 로그 확인용
- python/              : 내장 파이썬 및 PyTorch(CUDA) 라이브러리
- bin/                 : 고속 오디오 추출용 FFmpeg 내장 바이너리
- core/                : 프로그램 핵심 로직
- models/              : 음성인식 모델 저장 폴더 (최초 실행 시 자동 다운로드)

-------------------------------------------------------------------
[4] 문제 해결 (FAQ)
-------------------------------------------------------------------
Q. 브라우저가 자동으로 뜨지 않아요!
A. 웹 브라우저(크롬, 엣지 등)를 열고 주소창에 http://localhost:8501 을 입력하세요.

Q. 종료하고 싶어요!
A. 바탕에 열려 있는 검은색 콘솔(명령 프롬프트) 창을 닫으시면 완전히 종료됩니다.
===================================================================
"""
    (portable_root / "README_사용안내.txt").write_text(readme_content, encoding="utf-8")
    log("README_사용안내.txt 생성 완료")


def build_portable():
    t0 = time.time()
    log("=== 로컬 동영상 AI 자막 생성기 포터블 패키징 시작 ===")

    # 1. Clean / Prepare output directory
    if PORTABLE_DIR.exists():
        log(f"기존 임시 빌드 폴더 정리 중: {PORTABLE_DIR}")
        shutil.rmtree(PORTABLE_DIR)
    PORTABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Setup standalone Python runtime
    py_target = PORTABLE_DIR / "python"
    log(f"독립 파이썬 런타임 복사 중 ({PY_BASE} -> {py_target})...")
    py_target.mkdir(parents=True, exist_ok=True)

    # Binaries and DLLs
    for f in ["python.exe", "pythonw.exe", "python310.dll", "python3.dll", "vcruntime140.dll", "vcruntime140_1.dll"]:
        src_f = PY_BASE / f
        if src_f.exists():
            shutil.copy2(src_f, py_target / f)

    if (PY_BASE / "DLLs").exists():
        copy_tree_filtered(PY_BASE / "DLLs", py_target / "DLLs")

    # Standard Library
    lib_target = py_target / "Lib"
    lib_target.mkdir(parents=True, exist_ok=True)
    for item in (PY_BASE / "Lib").iterdir():
        if item.name in ("site-packages", "test"):
            continue
        if item.is_dir():
            copy_tree_filtered(item, lib_target / item.name)
        else:
            shutil.copy2(item, lib_target / item.name)

    # 3. Copy site-packages from .venv
    site_target = lib_target / "site-packages"
    log(f"의존성 패키지(PyTorch CUDA, Whisper, Streamlit 등) 복사 중 ({VENV_LIB} -> {site_target})...")
    copy_tree_filtered(VENV_LIB, site_target)

    # 4. Copy ffmpeg and ffprobe
    bin_target = PORTABLE_DIR / "bin"
    log(f"FFmpeg 바이너리 복사 중 -> {bin_target}...")
    ffmpeg_exe, ffprobe_exe = find_ffmpeg()
    bin_target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ffmpeg_exe, bin_target / "ffmpeg.exe")
    shutil.copy2(ffprobe_exe, bin_target / "ffprobe.exe")

    # 5. Copy Application Source Code
    log("프로그램 소스 코드 복사 중...")
    copy_tree_filtered(REPO_ROOT / "core", PORTABLE_DIR / "core")
    if (REPO_ROOT / "tools").exists():
        copy_tree_filtered(REPO_ROOT / "tools", PORTABLE_DIR / "tools")
    for f in ["app.py", "config.py", "requirements.txt", ".env.example"]:
        src_f = REPO_ROOT / f
        if src_f.exists():
            shutil.copy2(src_f, PORTABLE_DIR / f)

    # Models directory placeholder
    (PORTABLE_DIR / "models").mkdir(exist_ok=True)

    # 6. Create Batch scripts and README
    create_batch_files(PORTABLE_DIR)
    create_readme(PORTABLE_DIR)

    # Calculate uncompressed size
    uncompressed_bytes = sum(f.stat().st_size for f in PORTABLE_DIR.rglob("*") if f.is_file())
    uncompressed_gb = uncompressed_bytes / (1024**3)
    log(f"✅ 포터블 폴더 구성 완료: {PORTABLE_DIR} (총 {uncompressed_gb:.2f} GB)")

    # 7. Compress into standard ZIP
    log(f"포터블 패키지 ZIP 압축 생성 시작 -> {ZIP_PATH} (Deflate 압축)...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    file_list = [f for f in PORTABLE_DIR.rglob("*") if f.is_file()]
    total_files = len(file_list)
    last_print = time.time()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for idx, file_path in enumerate(file_list, start=1):
            arcname = file_path.relative_to(DIST_DIR)
            zf.write(file_path, arcname=arcname)
            if time.time() - last_print > 5.0 or idx == total_files:
                pct = (idx / total_files) * 100.0
                print(f"[*] 압축 진행률: {pct:5.1f}% ({idx}/{total_files} 파일)", flush=True)
                last_print = time.time()

    zip_bytes = ZIP_PATH.stat().st_size
    zip_gb = zip_bytes / (1024**3)
    elapsed = time.time() - t0

    print("=" * 65)
    print(f"🎉 포터블 패키징 및 ZIP 생성 완료!")
    print(f"📁 포터블 폴더: {PORTABLE_DIR} ({uncompressed_gb:.2f} GB)")
    print(f"📦 최종 ZIP 파일: {ZIP_PATH} ({zip_gb:.2f} GB)")
    print(f"⏱️ 소요 시간: {elapsed:.1f}초")
    print("=" * 65)


if __name__ == "__main__":
    build_portable()
