# GoSales Engine - Streamlit Launcher Script
# This script sets up the environment and launches the Streamlit UI

Write-Host "Starting GoSales Engine Streamlit UI..." -ForegroundColor Green

# Set Python path to include project root
$env:PYTHONPATH = "$PWD"

# Install required packages for Mermaid diagram support
Write-Host "Installing Mermaid diagram support..." -ForegroundColor Magenta
python -m pip install streamlit-mermaid

# Launch Streamlit
Write-Host "Setting PYTHONPATH to: $env:PYTHONPATH" -ForegroundColor Yellow
Write-Host "Starting Streamlit on http://localhost:8501" -ForegroundColor Cyan
streamlit run gosales/ui/app.py