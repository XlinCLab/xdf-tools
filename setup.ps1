# Usage: .\setup.ps1 [-Extras <extras>]
#   -Extras  comma-separated optional extras to install (default: audio,plot,test);
#            pass "" for the core install only
# Set $env:PYTHON to choose the interpreter, e.g. $env:PYTHON = "py -3.12"
param(
    [string]$Extras = "audio,plot,test"
)
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

# Split so that e.g. "py -3.12" works as the interpreter command
$pythonCmd, $pythonArgs = $python -split " "
& $pythonCmd @pythonArgs -c "import sys; sys.exit(sys.version_info < (3, 10))"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python 3.10+ is required (using: $python)"
    exit 1
}

if (-not (Test-Path .venv)) {
    Write-Host "Creating virtual environment in .venv ..."
    & $pythonCmd @pythonArgs -m venv .venv
} else {
    Write-Host "Reusing existing virtual environment in .venv"
}

$venvPython = ".venv\Scripts\python.exe"
& $venvPython -m pip install -U pip
if ($Extras) {
    & $venvPython -m pip install -e ".[$Extras]"
} else {
    & $venvPython -m pip install -e .
}

Write-Host ""
Write-Host "Done. Activate the environment with:"
Write-Host ""
Write-Host "    .venv\Scripts\Activate.ps1"
