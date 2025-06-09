Write-Host "Starting Glassbox LLM Debugger..." -ForegroundColor Green

Write-Host ""
Write-Host "========================================"
Write-Host "Starting Backend Server (Port 8000)..." -ForegroundColor Yellow
Write-Host "========================================"

Start-Process PowerShell -ArgumentList "-NoExit", "-Command", "cd backend; python app.py"

Start-Sleep -Seconds 3

Write-Host ""
Write-Host "========================================"
Write-Host "Starting Frontend Server (Port 3000)..." -ForegroundColor Cyan
Write-Host "========================================"

Start-Process PowerShell -ArgumentList "-NoExit", "-Command", "cd frontend; npm start"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Both servers are starting!" -ForegroundColor Green
Write-Host "Backend: http://localhost:8000" -ForegroundColor Yellow
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to close this window..."
Read-Host 