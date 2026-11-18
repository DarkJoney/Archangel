# Start omlx-intel-helper on Windows (a dev box; the target system is Ubuntu).
#   .\run-helper.ps1 [-Name my-intel-pc] [-Pair MAC_ADDRESS[:PORT]]
# Needs the project venv (..\..\..\.venv) with torch (XPU), numpy, psutil, zeroconf,
# safetensors installed. Windows firewall: allow TCP 8850-8852 from the Mac.
param(
    [string]$Name = $env:COMPUTERNAME.ToLower(),
    [string]$Pair = "",
    [string]$StateDir = "$env:USERPROFILE\.omlx-intel-helper",
    # Where the Mac's model copies go (default: <state dir>\models).
    [string]$ModelsDir = "",
    # Leave RAM to Windows; the iGPU shares it.
    [int]$OsReserveGiB = 6
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$python = Join-Path $root "..\.venv\Scripts\python.exe"
$env:OMLX_HELPER_OS_RESERVE_GIB = "$OsReserveGiB"
$env:PYTHONPATH = "$root"
$arguments = @("-m", "omlx_intel_helper.cli", "--state-dir", $StateDir, "serve", "--name", $Name)
if ($Pair) { $arguments += @("--pair", $Pair) }
if ($ModelsDir) { $arguments += @("--models-dir", $ModelsDir) }
Write-Host "starting: $python $($arguments -join ' ')"
& $python @arguments
