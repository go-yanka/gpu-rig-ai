# CBIC Verify pilot - run everything on this Windows desktop (no rig needed).
#
#   powershell -ExecutionPolicy Bypass -File pilot\local\run_local.ps1
#   powershell -ExecutionPolicy Bypass -File pilot\local\run_local.ps1 -Model qwen3:14b -Dense
#
# Needs: Python 3.10+, Ollama (https://ollama.com/download), and the corpus export
# (pilot\local\export_chunks.py run once on the rig) copied to -Data.
param(
  [string]$Data = "D:\_gpu_rig_ai\pilot_data",
  [string]$Model = "qwen3:8b",          # qwen3:14b if the GPU has >= 12 GB, qwen3:4b for CPU-only
  [switch]$Dense,                        # also build the BGE-M3 meaning index via Ollama (better recall; slow on CPU)
  [string]$LlmUrl = "http://127.0.0.1:11434",  # LM Studio: http://127.0.0.1:1234
  [int]$Port = 9600,
  [int]$SearchPort = 9601
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$Py = "python"

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

Step "Python packages"
& $Py -m pip install -q -r "$Repo\pilot\requirements.txt"

$UsingOllama = $LlmUrl -like "*11434*"
if ($UsingOllama) {
  Step "Ollama"
  try { Invoke-RestMethod "$LlmUrl/api/tags" | Out-Null }
  catch { throw "Ollama is not running at $LlmUrl. Install it from https://ollama.com/download and start it." }
  & ollama pull $Model
  # Ollama's default context is too small for 8 sources (~5K tokens) and truncates silently.
  $Local = "cbic-" + ($Model -replace "[:/]", "-")
  "FROM $Model`nPARAMETER num_ctx 16384`nPARAMETER temperature 0" | Set-Content "$env:TEMP\cbic.Modelfile"
  & ollama create $Local -f "$env:TEMP\cbic.Modelfile"
  $Model = $Local
  if ($Dense) { & ollama pull bge-m3 }
} elseif ($Dense) { throw "-Dense needs Ollama for the bge-m3 embeddings (it can run alongside LM Studio)." }

Step "Search index"
if (-not (Test-Path "$Data\SHA256SUMS")) { throw "No corpus export in $Data. Run pilot\local\export_chunks.py on the rig first (see pilot\README.md)." }
if (-not (Test-Path "$Data\search.sqlite") -or ($Dense -and -not (Test-Path "$Data\dense.npy"))) {
  $idx = @("$Repo\pilot\local\local_retrieve.py", "index", "--data", $Data)
  if ($Dense) { $idx += "--dense" }
  & $Py @idx
}
if (-not (Test-Path "$Data\amendment_graph.sqlite")) {
  & $Py "$Repo\rag\cbic_rag\amendment_graph.py" build --jsonl $Data --out "$Data\amendment_graph.sqlite"
}

Step "Tester keys"
$Tokens = "$Data\pilot_tokens.json"
if (-not (Test-Path $Tokens)) {
  $key = [guid]::NewGuid().ToString("N")
  "{ `"owner`": `"$key`" }" | Set-Content $Tokens
  Write-Host "Created $Tokens - add one line per tester: `"name@firm`": `"<random key>`""
}
$ownerKey = (Get-Content $Tokens | ConvertFrom-Json).owner
$AdminFile = "$Data\admin_token.txt"
if (-not (Test-Path $AdminFile)) { [guid]::NewGuid().ToString("N") | Set-Content $AdminFile }

Step "Starting local search on :$SearchPort"
$env:OLLAMA_URL = $LlmUrl
$search = Start-Process $Py -ArgumentList "`"$Repo\pilot\local\local_retrieve.py`" serve --data `"$Data`" --port $SearchPort" -PassThru -WindowStyle Minimized

Step "Starting pilot on :$Port (model: $Model)"
$env:CBIC_RAG_DIR = "$Repo\rag\cbic_rag;$Repo\cbic_rag"
$env:UPSTREAM_URL = "http://127.0.0.1:$SearchPort"
$env:LLM_URL = $LlmUrl
$env:LLM_MODEL = $Model
$env:LLM_KEY = "local"
$env:GRAPH_DB = "$Data\amendment_graph.sqlite"
$env:PILOT_DB = "$Data\pilot_log.sqlite"
$env:PILOT_TOKENS = $Tokens
$env:PILOT_ADMIN_TOKEN = (Get-Content $AdminFile).Trim()
$env:PILOT_MAX_CONCURRENT = "1"   # one local model = one answer at a time
Write-Host "`nOpen:   http://localhost:$Port/?t=$ownerKey"
Write-Host "Export: Invoke-WebRequest http://localhost:$Port/api/export.csv -Headers @{'X-Admin-Token'=(Get-Content '$AdminFile')} -OutFile feedback.csv"
Write-Host "Share with testers: cloudflared tunnel --url http://localhost:$Port   (then send https://<url>/?t=<their key>)`n"
try {
  Push-Location "$Repo\pilot\backend"
  & $Py -m uvicorn pilot_api:app --host 127.0.0.1 --port $Port
} finally {
  Pop-Location
  Stop-Process -Id $search.Id -ErrorAction SilentlyContinue
}
