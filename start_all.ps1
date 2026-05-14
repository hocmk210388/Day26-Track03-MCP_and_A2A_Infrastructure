# Windows helper: same order as start_all.sh (Git Bash / Linux).
# PowerShell does NOT run .sh files as bash — use this script, or: bash ./start_all.sh
#
# From repo root:
#   .\start_all.ps1
#
# Then in this terminal (or another), with same venv:
#   .\.venv\Scripts\python.exe test_client.py

$ErrorActionPreference = "Stop"
$Repo = $PSScriptRoot
Set-Location $Repo

$PyExe = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $PyExe)) {
    $PyExe = (Get-Command python -ErrorAction Stop).Source
}

function Start-A2AService {
    param(
        [Parameter(Mandatory = $true)][string]$Module
    )
    $command = "Set-Location `"$Repo`"; Write-Host `"Starting $Module ...`" -ForegroundColor Cyan; & `"$PyExe`" -m $Module"
    Start-Process powershell -ArgumentList @("-NoExit", "-NoProfile", "-Command", $command) | Out-Null
}

Write-Host "Launching each service in its own window. Close a window to stop that service." -ForegroundColor Green
Write-Host "Using: $PyExe" -ForegroundColor DarkGray
Write-Host ""

Start-A2AService -Module "registry"
Start-Sleep -Seconds 2

Start-A2AService -Module "tax_agent"
Start-A2AService -Module "compliance_agent"
Start-Sleep -Seconds 3

Start-A2AService -Module "law_agent"
Start-Sleep -Seconds 3

Start-A2AService -Module "customer_agent"

Write-Host ""
Write-Host "Ports: Registry 10000 | Customer 10100 | Law 10101 | Tax 10102 | Compliance 10103" -ForegroundColor Green
Write-Host "Next, run the client (new or this window):" -ForegroundColor Yellow
Write-Host "  & `"$PyExe`" `"$Repo\test_client.py`"" -ForegroundColor White
Write-Host ""
