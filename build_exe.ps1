# Builds DecisionDashboard.exe in the project folder.
# The exe contains only the dashboard window. The models still run with .venv,
# so keep the exe inside this folder. Close the dashboard before rebuilding.
#
# Run from PowerShell:  powershell -ExecutionPolicy Bypass -File build_exe.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Get-Process DecisionDashboard -ErrorAction SilentlyContinue) {
    Write-Host "DecisionDashboard.exe is running. Close it, then run this script again." -ForegroundColor Red
    exit 1
}

.\.venv\Scripts\python.exe -m pip install --quiet pyinstaller
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --onefile --windowed `
    --name DecisionDashboard --distpath . --workpath build --specpath build dashboard.py
$code = $LASTEXITCODE
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
if ($code -ne 0) {
    Write-Host "Build failed (exit code $code)." -ForegroundColor Red
    exit $code
}

Write-Host "Built $PSScriptRoot\DecisionDashboard.exe" -ForegroundColor Green
