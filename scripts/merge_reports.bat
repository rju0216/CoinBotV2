@echo off
REM ============================================================
REM  Merge yearly backtest reports into a single compounded report
REM
REM  Usage:
REM    merge_reports.bat [tag] [config_name]
REM
REM  Examples:
REM    merge_reports.bat 260418 config
REM    merge_reports.bat 260418 config_260418_CoinBot_TrendFollowing_V01
REM ============================================================

cd /d "%~dp0.."

if "%~1"=="" goto :USAGE
if "%~2"=="" goto :USAGE

python scripts/merge_yearly_reports.py --tag %~1 --config-name %~2
pause
exit /b 0

:USAGE
echo.
echo  [ERROR] tag and config_name are both required.
echo.
echo  Usage: %~nx0 [tag] [config_name]
echo  Example: %~nx0 260418 config_260418_CoinBot_TrendFollowing_V01
echo.
echo  Listing available reports...
echo.
python scripts/merge_yearly_reports.py --tag _ --config-name _
pause
exit /b 1
