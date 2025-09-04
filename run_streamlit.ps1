# GoSales Engine - Streamlit Launcher Script
# This script sets up the environment and launches the Streamlit UI

Write-Host "Starting GoSales Engine Streamlit UI..." -ForegroundColor Green

# Set Python path to include project root
$env:PYTHONPATH = "$PWD"

# Check if streamlit-mermaid is installed, install if needed
Write-Host "Checking Mermaid diagram support..." -ForegroundColor Magenta
python -c "import streamlit_mermaid" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing streamlit-mermaid..." -ForegroundColor Yellow
    python -m pip install streamlit-mermaid
} else {
    Write-Host "streamlit-mermaid already installed." -ForegroundColor Green
}

# Launch Streamlit
Write-Host "Setting PYTHONPATH to: $env:PYTHONPATH" -ForegroundColor Yellow
Write-Host "Starting Streamlit on http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
streamlit run gosales/ui/app.py