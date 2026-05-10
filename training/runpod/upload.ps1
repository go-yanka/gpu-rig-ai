# upload.ps1 — push training data + scripts from Windows laptop to the RunPod pod.
# Prefers runpodctl if a pod ID is given; else falls back to scp.
#
# Usage:
#   .\upload.ps1 -PodIp 1.2.3.4 -PodPort 12345
#   .\upload.ps1 -PodId abc123xyz          # runpodctl path (if installed)

[CmdletBinding()]
param(
    [string]$PodIp,
    [int]   $PodPort,
    [string]$PodId,
    [string]$SshKey = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$RemoteDir = "/workspace/cbic"
)

$ErrorActionPreference = "Stop"
$local = "D:\_gpu_rig_ai\training"
$runpodDir = Join-Path $local "runpod"

$files = @(
    (Join-Path $local     "curated_pairs.jsonl"),
    (Join-Path $local     "hard_negatives.jsonl"),
    (Join-Path $local     "finetune_bge_m3.py"),
    (Join-Path $local     "prep_pairs.py"),
    (Join-Path $runpodDir "setup.sh"),
    (Join-Path $runpodDir "run_training.sh")
)

# Pre-flight: all files must exist
$missing = $files | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Host "[upload] MISSING FILES:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

# Optional gold set for IR eval (won't fail if absent)
$goldPath = Join-Path $local "gold.jsonl"
if (Test-Path $goldPath) { $files += $goldPath }

Write-Host "[upload] files:" -ForegroundColor Cyan
$files | ForEach-Object { Write-Host "  $_" }

# --- runpodctl path ---
if ($PodId) {
    $ctl = Get-Command runpodctl -ErrorAction SilentlyContinue
    if (-not $ctl) {
        Write-Host "[upload] runpodctl not on PATH; install or use -PodIp/-PodPort" -ForegroundColor Red
        exit 1
    }
    foreach ($f in $files) {
        Write-Host "[upload] runpodctl send $f -> $PodId"
        runpodctl send $f
        Write-Host "  (on pod, run: runpodctl receive <code>)"
    }
    Write-Host "[upload] done via runpodctl. Move files on pod to $RemoteDir manually." -ForegroundColor Green
    exit 0
}

# --- scp path ---
if (-not $PodIp -or -not $PodPort) {
    Write-Host "[upload] need -PodIp AND -PodPort (or -PodId for runpodctl)" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $SshKey)) {
    Write-Host "[upload] SSH key not found: $SshKey" -ForegroundColor Red
    exit 1
}

# Ensure remote dir exists
& ssh -i $SshKey -p $PodPort -o StrictHostKeyChecking=no "root@$PodIp" "mkdir -p $RemoteDir"
if ($LASTEXITCODE -ne 0) { Write-Host "[upload] ssh mkdir failed" -ForegroundColor Red; exit 1 }

foreach ($f in $files) {
    $name = Split-Path -Leaf $f
    Write-Host "[upload] scp $name -> root@${PodIp}:$RemoteDir/"
    & scp -i $SshKey -P $PodPort -o StrictHostKeyChecking=no $f "root@${PodIp}:$RemoteDir/$name"
    if ($LASTEXITCODE -ne 0) { Write-Host "[upload] scp failed for $name" -ForegroundColor Red; exit 1 }
}

# Make shell scripts executable on the pod
& ssh -i $SshKey -p $PodPort "root@$PodIp" "chmod +x $RemoteDir/setup.sh $RemoteDir/run_training.sh"

Write-Host "[upload] done. On pod: cd $RemoteDir && bash setup.sh && bash run_training.sh" -ForegroundColor Green
