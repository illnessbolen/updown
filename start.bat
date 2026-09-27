@echo off
rem One-command launcher for Windows:
rem   start.bat          -> menu
rem   start.bat paper    -> run a mode directly (any bot.py arguments)
rem   start.bat test     -> run the test suite
rem First run creates .venv, installs dependencies and copies .env.example to .env.
setlocal
cd /d "%~dp0"
chcp 65001 >nul

set "PY="
where py >nul 2>nul
if errorlevel 1 goto try_python
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 goto try_python
set "PY=py -3"
goto have_py
:try_python
where python >nul 2>nul
if errorlevel 1 goto no_py
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 goto no_py
set "PY=python"
:have_py

if exist ".venv\Scripts\python.exe" goto have_venv
echo [setup] creating virtual environment .venv
%PY% -m venv .venv
if errorlevel 1 goto fail
:have_venv
set "VPY=.venv\Scripts\python.exe"

if exist ".venv\.installed" goto have_deps
echo [setup] installing dependencies (first run only)
"%VPY%" -m pip install --upgrade pip >nul
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail
type nul > ".venv\.installed"
:have_deps

if exist ".env" goto have_env
copy ".env.example" ".env" >nul
echo [setup] created .env from .env.example - edit it to change settings
:have_env

if "%~1"=="" goto menu
if /i "%~1"=="test" goto run_tests
"%VPY%" bot.py %*
exit /b %errorlevel%

:menu
echo.
echo   Polymarket Up/Down - what to run?
echo     1) discover    active Up/Down markets right now
echo     2) shadow      live data + model + signal log (no orders), records ticks
echo     3) paper       shadow + simulated execution on a virtual balance, records ticks
echo     4) analyze     model calibration and signal / paper results
echo     5) stats       statistics of settled trades
echo     6) hypothesis  is the edge distinguishable from zero?
echo     7) report      weekly report now
echo     8) test        run the tests
echo     0) exit
echo   Stop shadow/paper with Ctrl+C. Emergency stop of orders: create a file named STOP here
echo   (type nul ^> STOP).
echo.
set "choice="
set /p "choice=Choice: "
if "%choice%"=="1" "%VPY%" bot.py discover
if "%choice%"=="2" "%VPY%" bot.py shadow --record
if "%choice%"=="3" "%VPY%" bot.py paper --record
if "%choice%"=="4" "%VPY%" bot.py analyze
if "%choice%"=="5" "%VPY%" bot.py stats
if "%choice%"=="6" "%VPY%" bot.py hypothesis
if "%choice%"=="7" "%VPY%" bot.py report
if "%choice%"=="8" goto run_tests
exit /b 0

:run_tests
"%VPY%" -m pip install -q -r requirements-dev.txt
"%VPY%" -m pytest -q
exit /b %errorlevel%

:no_py
echo Python 3.10 or newer is required (python --version). Install it from python.org
echo and tick "Add python.exe to PATH", then run start.bat again.
exit /b 1

:fail
echo Setup failed - see the messages above.
exit /b 1
