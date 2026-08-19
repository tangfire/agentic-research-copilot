param(
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

$env:ARC_STORAGE_PATH = ".arc/workbench-local-$Port.db"
$env:ARC_LANGGRAPH_CHECKPOINT_PATH = ".arc/langgraph-workbench-local-$Port.sqlite"
$env:ARC_STRICT_PROVIDERS = "false"
$env:ARC_JOB_QUEUE_BACKEND = "in_process"
$env:ARC_MCP_ENABLED = "false"
$env:ARC_MCP_SERVER_URL = ""
$env:ARC_MCP_TOOLS = ""
$env:ARC_MCP_PROMPT = ""
$env:ARC_MCP_AUTH_REQUIRED = "false"
$env:ARC_MCP_AUTH_TOKEN = ""
$env:ARC_QDRANT_URL = ""
$env:ARC_QDRANT_PREFER_LOCAL = "true"
$env:ARC_QDRANT_LOCATION = ".arc/qdrant-workbench-$Port"

$healthUrl = "http://127.0.0.1:$Port/health"
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existing) {
  $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
  [pscustomobject]@{
    ok = $health.status -eq "ok"
    pid = $existing.OwningProcess
    url = "http://127.0.0.1:$Port/"
    health = $healthUrl
    reused = $true
  } | ConvertTo-Json -Compress
  exit 0
}

$args = @(
  "-m",
  "uvicorn",
  "agentic_research_copilot.server:create_app",
  "--factory",
  "--host",
  "127.0.0.1",
  "--port",
  "$Port"
)

$process = Start-Process -FilePath python.exe -ArgumentList $args -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru

$health = $null
for ($i = 0; $i -lt 30; $i++) {
  try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
    break
  } catch {
    Start-Sleep -Seconds 1
  }
}

if (-not $health) {
  throw "Workbench did not become healthy at $healthUrl within 30 seconds. Process id: $($process.Id)"
}

[pscustomobject]@{
  ok = $health.status -eq "ok"
  pid = $process.Id
  url = "http://127.0.0.1:$Port/"
  health = $healthUrl
  reused = $false
} | ConvertTo-Json -Compress
