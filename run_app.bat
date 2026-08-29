@echo off
chcp 65001 > nul
title Local Video Subtitle Generator

echo ================================================================
echo   Local Video Subtitle Generator
echo ================================================================
echo.

cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\streamlit.exe" (
    "%~dp0.venv\Scripts\streamlit.exe" run app.py
) else (
    echo [ERROR] .venv virtual environment not found.
    pause
)
