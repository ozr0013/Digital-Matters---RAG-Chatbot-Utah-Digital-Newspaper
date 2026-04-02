@echo off
title Utah Digital Newspapers RAG Chatbot Launcher
set "PROJECT_DIR=C:\Users\u1529771\Desktop\ORAG\Digital-Matters---RAG-Chatbot-Utah-Digital-Newspaper"

echo ============================================
echo  Utah Digital Newspapers RAG Chatbot
echo ============================================
echo.
echo Starting server...
echo NOTE: First run builds the full search index and may take
echo       several minutes depending on your dataset size.
echo.

:: Check if server is already running on port 5000
netstat -ano | findstr ":5000 " | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo Server already running - opening browser...
    start "" "http://localhost:5000"
    timeout /t 2 /nobreak >nul
    exit
)

:: Start Flask server in a new terminal window (stays open to show logs)
start "Utah Newspapers RAG Chatbot - Server" cmd /k "cd /d "%PROJECT_DIR%" && call venv\Scripts\activate.bat && python app.py"

:: Wait for Flask to initialize (extra time for full index build on first run)
echo Waiting for server to start...
timeout /t 12 /nobreak >nul

:: Open browser
echo Opening browser at http://localhost:5000
start "" "http://localhost:5000"

exit
