# push.ps1 — run from Windows when SSH to rig is working again.
# Copies D:\_gpu_rig_ai\cbic_rag to the rig and runs deploy.sh.
#
# If SSH is stuck from a previous session, first restart sshd on the rig
# via the web terminal / console:
#   sudo systemctl restart ssh

$ErrorActionPreference = 'Stop'
$src = 'D:\_gpu_rig_ai\cbic_rag'
$dst = 'rig:/opt/indian-legal-ai/rag/cbic_rag'
$srcSpec = 'D:\_gpu_rig_ai\reingest_spec'
$dstSpec = 'rig:/opt/indian-legal-ai/reingest_spec'

Write-Host "[1/4] Testing SSH..." -ForegroundColor Cyan
$t = ssh -o ConnectTimeout=15 -o BatchMode=yes rig "echo ok" 2>&1
if ($t -ne 'ok') {
    Write-Host "SSH not working: $t" -ForegroundColor Red
    Write-Host "On the rig, run:   sudo systemctl restart ssh" -ForegroundColor Yellow
    exit 1
}

Write-Host "[2/4] Pushing cbic_rag (incl. static/, api_v2_shadow, hyde)..." -ForegroundColor Cyan
ssh rig "mkdir -p /opt/indian-legal-ai/rag/cbic_rag/static"
scp -r "$src\*" "$dst/"

Write-Host "[3/4] Pushing reingest_spec (v2 chunker/ingest/evaluators/scripts/eval/)..." -ForegroundColor Cyan
ssh rig "mkdir -p /opt/indian-legal-ai/reingest_spec/evaluators /opt/indian-legal-ai/reingest_spec/eval"
# exclude __pycache__, runs, result jsons to keep rsync quick
scp -r "$srcSpec\chunker_v2.py" "$srcSpec\ingest_v2.py" "$srcSpec\topic_tagger.py" `
       "$srcSpec\text_source_detector.py" "$srcSpec\check_lessons.py" `
       "$srcSpec\theta_tune.py" "$srcSpec\archive_v1.sh" "$srcSpec\rollback_v1.sh" `
       "$srcSpec\snapshot_v2.sh" "$srcSpec\LESSONS_APPLIED.md" "$srcSpec\SPEC.md" "$srcSpec\RUNBOOK.md" `
       "$dstSpec/"
scp -r "$srcSpec\evaluators\*.py" "$srcSpec\evaluators\*.sh" "$dstSpec/evaluators/"
scp -r "$srcSpec\eval\*.json" "$dstSpec/eval/"

Write-Host "[4/4] Deploying..." -ForegroundColor Cyan
ssh rig "chmod +x /opt/indian-legal-ai/rag/cbic_rag/deploy.sh /opt/indian-legal-ai/reingest_spec/*.sh /opt/indian-legal-ai/reingest_spec/evaluators/*.sh && bash /opt/indian-legal-ai/rag/cbic_rag/deploy.sh"

Write-Host ""
Write-Host "Monitor ingestion:    ssh rig screen -r cbic-ingest" -ForegroundColor Green
Write-Host "API health check:     curl http://192.168.1.107:9500/health" -ForegroundColor Green
Write-Host "Qdrant cbic collection: curl http://192.168.1.107:6343/collections/cbic_v1" -ForegroundColor Green
