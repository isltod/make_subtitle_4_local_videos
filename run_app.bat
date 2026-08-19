@echo off
chcp 65001 > nul
title Local Video Subtitle Generator (AI 자막 생성기)
echo ================================================================
echo 🎬 Local Video Subtitle Generator 를 실행합니다...
echo (RTX 3090 Whisper + Gemini 3.7 Flash)
echo ================================================================

cd /d "%~dp0"
if exist ".\.venv\Scripts\streamlit.exe" (
    ".\.venv\Scripts\streamlit.exe" run app.py
) else (
    echo [ERROR] .venv 가상환경이 발견되지 않았습니다.
    pause
)
