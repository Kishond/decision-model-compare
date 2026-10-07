# Builds DecisionDashboard.exe in the project folder. Works on a fresh computer:
# if .venv doesn't exist yet, it is created first (Python 3.10-3.13 must be installed).
# The exe contains only the dashboard window; its Setup button installs the rest.
#
# Easiest: double-click Build-Dashboard.bat
# Or run:  powershell -ExecutionPolicy Bypass -File build_exe.ps1

param([switch]$NoPause)  # -NoPause: don't wait for Enter at the end (for automated runs)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$PythonDownload = "https://www.python.org/downloads/windows/"

function Finish([int]$Code, [string]$Message, [string]$Color) {
    Write-Host ""
    Write-Host $Message -ForegroundColor $Color
    # Keep the window open so the message can be read.
    if (-not $NoPause) { Read-Host "Press Enter to close" | Out-Null }
    exit $Code
}

function Find-Python {
    # Windows "py" launcher first, newest supported version first.
    foreach ($version in "3.13", "3.12", "3.11", "3.10") {
        try {
            $exe = & py "-$version-64" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        } catch {}
    }
    # Then python on PATH, if it's a supported 64-bit version.
    $check = "import sys, struct; ok = (3, 10) <= sys.version_info[:2] <= (3, 13) and struct.calcsize('P') == 8; print(sys.executable if ok else '')"
    foreach ($name in "python", "python3") {
        try {
            $exe = & $name -c $check 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        } catch {}
    }
    return $null
}

try {
    if (Get-Process DecisionDashboard -ErrorAction SilentlyContinue) {
        Finish 1 "DecisionDashboard.exe is running. Close it, then run this again." Red
    }

    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Host "No Python environment (.venv) yet. Looking for Python 3.10-3.13 (64-bit)..."
        $basePython = Find-Python
        if (-not $basePython) {
            Finish 1 ("Python 3.10-3.13 (64-bit) was not found on this computer.`n" +
                      "Install Python 3.13 from $PythonDownload`n" +
                      "and tick 'Add python.exe to PATH' in the installer. Then run this again.") Red
        }
        Write-Host "Found $basePython"
        Write-Host "Creating .venv..."
        & $basePython -m venv .venv
        if ($LASTEXITCODE -ne 0) { Finish 1 "Creating .venv failed." Red }
    }

    Write-Host "Installing PyInstaller (the tool that builds the exe)..."
    & $venvPython -m pip install --quiet --disable-pip-version-check pyinstaller
    if ($LASTEXITCODE -ne 0) { Finish 1 "Installing PyInstaller failed. Check your internet connection." Red }

    Write-Host "Building DecisionDashboard.exe (about a minute)..."
    & $venvPython -m PyInstaller --noconfirm --onefile --windowed --log-level WARN `
        --name DecisionDashboard --distpath . --workpath build --specpath build dashboard.py
    $code = $LASTEXITCODE
    Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
    if ($code -ne 0) { Finish $code "Build failed (exit code $code). See the messages above." Red }

    Finish 0 ("Built $PSScriptRoot\DecisionDashboard.exe`n" +
              "Double-click it. The first time, it offers to run Setup, which installs the rest.") Green
} catch {
    Finish 1 "Something went wrong: $_" Red
}
