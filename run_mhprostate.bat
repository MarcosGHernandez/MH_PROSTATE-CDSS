@echo off
setlocal
title MH PROSTATE-CDSS (Clinical Decision Support System)
color 0B

echo.
echo  =============================================================
echo  *                                                           *
echo  *   $$\      $$\ $$\   $$\       $$$$$$$\   $$$$$$\   $$$$$$\  *
echo  *   $$$\    $$$ ^|$$ ^|  $$ ^|      $$  __$$\ $$  __$$\ $$  __$$\ *
echo  *   $$$$\  $$$$ ^|$$ ^|  $$ ^|      $$ ^|  $$ ^|$$ /  \__^|$$ /  \__^|*
echo  *   $$\$$\$$ $$ ^|$$$$$$$$ ^|      $$$$$$$  ^|\$$$$$$\   \$$$$$$\  *
echo  *   $$ \$$$  $$ ^|$$  __$$ ^|      $$  ____/  \____$$\   \____$$\ *
echo  *   $$ ^|\$  / $$ ^|$$ ^|  $$ ^|      $$ ^|      $$\   $$ ^|$$\   $$ ^|*
echo  *   $$ ^| \_/  $$ ^|$$ ^|  $$ ^|      $$ ^|      \$$$$$$  ^|\$$$$$$  ^|*
echo  *   \__^|     \__^|\__^|  \__^|      \__^|       \______/  \______/ *
echo  *                                                           *
echo  *             MH PROSTATE-CDSS (VERSION 1.0)                *
echo  *                ZERO-EGRESS - EBM 2026                     *
echo  =============================================================
echo.

:: --- Process Cleanup ---
echo [CLEANUP] Closing existing instances...
powershell -Command "Stop-Process -Name 'python' -ErrorAction SilentlyContinue" >nul 2>&1
powershell -Command "Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue ^| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

:: --- Environment Check ---
if exist .venv\Scripts\activate (
    echo [ENV] Virtual environment discovered.
    call .venv\Scripts\activate
) else (
    echo [ENV] Using system Python.
)

:: --- Launch Sequence ---
echo [SYSTEM] Initializing MH PROSTATE-CDSS...
set PYTHONPATH=src/vision;%PYTHONPATH%
start /B python -m streamlit run src/ui/dashboard.py --server.port 8501 --server.headless true

echo.
echo [READY] System active at: http://localhost:8501
echo [NOTE] Maintain this window open during diagnostic session.
echo.

timeout /t 5 > nul
start http://localhost:8501
pause
exit
