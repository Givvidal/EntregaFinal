$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" } 

Write-Host "[1/3] Comprobando Redis (solo se usa para la cola y estados)..."
& $python scripts/check_redis.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Redis no responde o faltan claves en .env. Revisá la configuración y volvé a ejecutar run.ps1." -ForegroundColor Red
    exit 1
}

Write-Host "[2/3] Iniciando Phoenix en una ventana separada (observabilidad)..."
try {
    Start-Process -FilePath $python -ArgumentList @("-m", "phoenix.server.main", "serve") -WorkingDirectory $PSScriptRoot -WindowStyle Minimized
    Start-Sleep -Seconds 3
} catch {
    Write-Warning "No se pudo iniciar Phoenix automáticamente. Podés iniciarlo por separado con: phoenix serve"
}

Write-Host "[3/3] Iniciando FastAPI en http://127.0.0.1:8000/docs"
& $python -m uvicorn main:app --reload
