@echo off
chcp 65001 > nul
title RTX 3090 Power Limit (280W)
cd /d "%~dp0"

net session >nul 2>&1
if errorlevel 1 (
    echo [*] Requesting Administrator privileges...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs" 
    exit /b
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0set_rtx3090_power_limit.ps1"
