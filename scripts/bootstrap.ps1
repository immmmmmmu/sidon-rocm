# Create a fresh local Python environment only. No downloads or driver changes.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'Run this script on the Windows target machine.'
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot '.venv'
if (Test-Path $VenvPath) {
    throw 'Existing .venv is preserved. Inspect it or use a separate checkout.'
}
& py -3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
& py -3.12 -m venv $VenvPath
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
Write-Output "Created $VenvPath. Follow README.md to install the pinned AMD packages."
