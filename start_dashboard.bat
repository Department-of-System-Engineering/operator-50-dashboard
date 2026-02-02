@echo off

call :start_if_not_running "C:\Program Files\Docker\Docker\Docker Desktop.exe"
call :start_if_not_running "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"

start "" powershell -NoExit -Command "cd chatbot\server; docker compose up –d db; cd ..\..; 'ready' | Out-File 'w1.ready'; cd chatbot\server; venv\Scripts\activate; uvicorn main:app --reload --host 127.0.0.1 --port 8000"
start "" powershell -NoExit -Command "cd chatbot\client\chatbot; npm i; cd ..\..\..; 'ready' | Out-File 'w2.ready'; cd chatbot\client\chatbot; npm start"

echo Waiting for Window1 and Window2 to signal readiness...

:wait
if not exist w1.ready goto wait
if not exist w2.ready goto wait

start "" powershell -Command "Start-Process msedge.exe '--new-window http://localhost:4200/home'"
start "" powershell -NoExit -Command "cd dash; venv\Scripts\activate; streamlit run .\app_redesigned.py;"

del /Q w1.ready
del /Q w2.ready

:is_running
REM %1 = Process name 
tasklist | find /i "%~1" >nul
exit /b %errorlevel%

:start_if_not_running
REM %1 = Full path to EXE
set "FULLPATH=%~1"
set "EXENAME=%~nx1"

call :is_running "%EXENAME%"
if %errorlevel%==0 (
    echo %EXENAME% is already running
) else (
    echo Starting %EXENAME%...
    start "" "%FULLPATH%"
)
exit /b