# GoSales Engine - Web Dashboard Launcher

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "GoSales Engine - Web Dashboard" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

# Check if Flask is installed
Write-Host "`nChecking dependencies..." -ForegroundColor Yellow
python -c "import flask" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Flask not found. Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host "`nStarting web dashboard server..." -ForegroundColor Green
Write-Host "Dashboard will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Gray
Write-Host "=" * 60 -ForegroundColor Cyan

# Start the server
python server.py

