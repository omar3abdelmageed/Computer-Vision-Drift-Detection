param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Expected virtual environment Python at $Python"
}

if (-not $SkipTests) {
    $TestTemp = Join-Path $ProjectRoot "build\pytest-$PID"
    & $Python -m pytest -q -p no:cacheprovider --basetemp $TestTemp
    if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
}

& $Python -m PyInstaller --noconfirm --clean packaging\CVMonitor.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

Write-Host "Desktop bundle created at dist\CVMonitor"
