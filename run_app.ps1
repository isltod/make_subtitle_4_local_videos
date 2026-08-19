$OutputEncoding = [System.Text.Encoding]::UTF8
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "🎬 Local Video Subtitle Generator 를 실행합니다..." -ForegroundColor Green
Write-Host "(RTX 3090 Whisper + Gemini 3.7 Flash)" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan

Set-Location $PSScriptRoot
if (Test-Path ".\.venv\Scripts\streamlit.exe") {
    & ".\.venv\Scripts\streamlit.exe" run app.py
} else {
    Write-Error "[ERROR] .venv 가상환경이 발견되지 않았습니다."
}
