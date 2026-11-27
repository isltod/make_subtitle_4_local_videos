$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  NVIDIA RTX 3090 전력 제한 안정화 (280W 설정)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$lines = nvidia-smi --query-gpu=index,name --format=csv,noheader 2>$null
$idx = $null
foreach ($line in $lines) {
    if ($line -match '^\s*(\d+)\s*,.*3090') {
        $idx = $matches[1]
        break
    }
}

if ($null -eq $idx) {
    Write-Host "[!] RTX 3090 장치를 nvidia-smi에서 찾지 못했습니다." -ForegroundColor Red
    Write-Host "[!] GPU가 유실된 상태라면 먼저 PC를 재부팅하세요." -ForegroundColor Yellow
    nvidia-smi
    Write-Host ""
    Read-Host "종료하려면 Enter 키를 누르세요"
    exit 1
}

Write-Host "[*] 감지된 RTX 3090 인덱스: $idx"
Write-Host "[*] 전력 제한을 280W로 설정합니다 (재부팅 시 기본값 350W로 초기화됩니다)..."
nvidia-smi -i $idx -pl 280

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] RTX 3090 전력 제한이 280W로 성공적으로 적용되었습니다!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[!] 전력 제한 설정에 실패했습니다. 드라이버 또는 관리자 권한을 확인하세요." -ForegroundColor Red
}

Write-Host ""
Read-Host "창을 닫으려면 Enter 키를 누르세요"
