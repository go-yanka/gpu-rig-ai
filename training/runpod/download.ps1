# download.ps1 — pull the fine-tuned model back to the laptop and extract it.
#
# Usage:
#   .\download.ps1 -PodIp 1.2.3.4 -PodPort 12345

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$PodIp,
    [Parameter(Mandatory=$true)][int]   $PodPort,
    [string]$SshKey = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$RemotePath = "/workspace/cbic/bge-m3-cbic-v1.tar.gz",
    [string]$LocalDir = "D:\_gpu_rig_ai\training"
)

$ErrorActionPreference = "Stop"
$tar    = Join-Path $LocalDir "bge-m3-cbic-v1.tar.gz"
$sha    = "$tar.sha256"
$outDir = Join-Path $LocalDir "bge-m3-cbic-v1"

Write-Host "[download] scp model tarball -> $tar" -ForegroundColor Cyan
& scp -i $SshKey -P $PodPort -o StrictHostKeyChecking=no "root@${PodIp}:$RemotePath"        $tar
if ($LASTEXITCODE -ne 0) { Write-Host "[download] scp failed" -ForegroundColor Red; exit 1 }
& scp -i $SshKey -P $PodPort -o StrictHostKeyChecking=no "root@${PodIp}:$RemotePath.sha256" $sha
if ($LASTEXITCODE -ne 0) { Write-Host "[download] sha256 fetch failed (non-fatal)" -ForegroundColor Yellow }

# Verify checksum (best-effort)
if (Test-Path $sha) {
    $expected = (Get-Content $sha | Select-Object -First 1).Split()[0]
    $actual   = (Get-FileHash $tar -Algorithm SHA256).Hash.ToLower()
    if ($expected -ne $actual) {
        Write-Host "[download] CHECKSUM MISMATCH: expected=$expected actual=$actual" -ForegroundColor Red
        exit 1
    }
    Write-Host "[download] sha256 OK" -ForegroundColor Green
}

# Extract (tar is bundled with Windows 10+)
if (Test-Path $outDir) {
    Write-Host "[download] removing existing $outDir" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $outDir
}
Write-Host "[download] extracting -> $outDir" -ForegroundColor Cyan
Push-Location $LocalDir
try {
    & tar -xzf (Split-Path -Leaf $tar)
    if ($LASTEXITCODE -ne 0) { throw "tar extract failed" }
} finally {
    Pop-Location
}

Write-Host "[download] done. Model at: $outDir" -ForegroundColor Green
Write-Host "[download] Next: deploy_to_rig.md"
