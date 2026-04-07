# Launch-Dashboard.ps1
# MHC Prostate-CDSS Dashboard Orchestrator
# Terminates existing instances and starts a fresh one.

$Port = 8501

# Find process on port 8501
$Process = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -First 1
if ($Process) {
    Write-Host "Cerrando dashboard en puerto $Port (PID $Process)..." -ForegroundColor Yellow
    Stop-Process -Id $Process -Force -ErrorAction SilentlyContinue
}

Write-Host "Iniciando MH Prostate-CDSS Dashboard..." -ForegroundColor Green
# Start in a new background job or window
Start-Process -FilePath "python" -ArgumentList "-m streamlit run src/ui/dashboard.py --server.port $Port --server.headless true" -WindowStyle Hidden -RedirectStandardOutput "streamlit_v12.log" -RedirectStandardError "streamlit_v12.log"

Write-Host "El Dashboard debería estar disponible pronto en http://localhost:$Port" -ForegroundColor Cyan
