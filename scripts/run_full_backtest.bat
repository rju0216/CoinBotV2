@echo off
setlocal EnableDelayedExpansion
REM ============================================================
REM  Full Backtest - Yearly Parallel Execution
REM  1) Download CSV candle data (sync, once via download_history.py)
REM  2) Launch 7 terminal windows for parallel backtest via
REM     `python -m src.main backtest --config ... --start ... --end ...`
REM
REM  Usage:
REM    run_full_backtest.bat config/default.yaml
REM    run_full_backtest.bat               (no args = config/default.yaml)
REM ============================================================

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

if "%~1"=="" (
    echo  No config argument specified.
    echo  Run backtest with default config/default.yaml?
    echo.
    set /p CONFIRM="  Continue? (y/n): "
    if /i "!CONFIRM!" NEQ "y" (
        echo.
        echo  Cancelled.
        echo  Usage: %~nx0 ^<config-path^>
        echo  Example: %~nx0 config/default.yaml
        pause
        exit /b 0
    )
    set "CONFIG=config/default.yaml"
) else (
    set "CONFIG=%~1"
)

if not exist "%CONFIG%" (
    echo [ERROR] File not found: %CONFIG%
    pause
    exit /b 1
)

echo ============================================================
echo  Config: %CONFIG%
echo ============================================================
echo.
echo  Phase 1: Download CSV candle data
echo  Timeframes: 1d, 4h, 15m (2020-01-01 ~ 2026-04-18)
echo  Skips download if CSV already covers the range.
echo.

python scripts\download_history.py --config %CONFIG% --timeframe 1d,4h,15m --start 2020-01-01 --end 2026-04-18

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Data download failed. Backtest will not start.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Phase 2: Launch 7 yearly backtests in parallel
echo ============================================================
echo.

start "BT 2020" cmd /k "cd /d !PROJECT_ROOT! && python -m src.main backtest --config !CONFIG! --start 2020-01-01 --end 2020-12-31 && echo. && echo [2020] DONE && pause"
start "BT 2021" cmd /k "cd /d !PROJECT_ROOT! && python -m src.main backtest --config !CONFIG! --start 2021-01-01 --end 2021-12-31 && echo. && echo [2021] DONE && pause"
start "BT 2022" cmd /k "cd /d !PROJECT_ROOT! && python -m src.main backtest --config !CONFIG! --start 2022-01-01 --end 2022-12-31 && echo. && echo [2022] DONE && pause"
start "BT 2023" cmd /k "cd /d !PROJECT_ROOT! && python -m src.main backtest --config !CONFIG! --start 2023-01-01 --end 2023-12-31 && echo. && echo [2023] DONE && pause"
start "BT 2024" cmd /k "cd /d !PROJECT_ROOT! && python -m src.main backtest --config !CONFIG! --start 2024-01-01 --end 2024-12-31 && echo. && echo [2024] DONE && pause"
start "BT 2025" cmd /k "cd /d !PROJECT_ROOT! && python -m src.main backtest --config !CONFIG! --start 2025-01-01 --end 2025-12-31 && echo. && echo [2025] DONE && pause"
start "BT 2026" cmd /k "cd /d !PROJECT_ROOT! && python -m src.main backtest --config !CONFIG! --start 2026-01-01 --end 2026-04-18 && echo. && echo [2026] DONE && pause"

echo  7 windows launched. Check each window for results.
echo  Results: data\backtest_reports\00_Working\
echo.
pause
