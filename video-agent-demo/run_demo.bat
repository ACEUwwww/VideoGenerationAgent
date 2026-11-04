@echo off
rem One-command demo for Windows. Mirrors run_demo.sh.
rem   run_demo.bat check     environment self-check (free)
rem   run_demo.bat test      run the test suite
rem   run_demo.bat serve     start the API on :8000
rem   run_demo.bat simulate  zero-cost mock run of a 3-shot episode
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Missing virtualenv. Run:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)

if "%~1"=="" goto check
if /i "%~1"=="check" goto check
if /i "%~1"=="test" goto test
if /i "%~1"=="serve" goto serve
if /i "%~1"=="simulate" goto simulate
echo Usage: run_demo.bat [check^|test^|simulate^|serve]
exit /b 1

:check
if not exist ".env" (
  echo Missing .env. Run: copy .env.example .env  then fill in the keys.
  exit /b 1
)
"%PY%" scripts\doctor.py
exit /b %errorlevel%

:test
"%PY%" -m pytest tests -q
exit /b %errorlevel%

:simulate
set IMAGE_PROVIDER=mock
set VIDEO_PROVIDER=mock
"%PY%" -m scripts.simulate
exit /b %errorlevel%

:serve
echo Backend: http://localhost:8000
echo Frontend: open another terminal, cd frontend, npm run dev
"%PY%" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
exit /b %errorlevel%
