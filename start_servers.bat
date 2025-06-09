@echo off
echo Starting Glassbox LLM Debugger...

echo.
echo ========================================
echo Starting Backend Server (Port 8000)...
echo ========================================
start cmd /k "cd backend && python app.py"

timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo Starting Frontend Server (Port 3000)...
echo ========================================
start cmd /k "cd frontend && npm start"

echo.
echo ========================================
echo Both servers are starting!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo ========================================
echo.
echo Press any key to close this window...
pause >nul 