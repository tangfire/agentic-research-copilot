param(
  [int]$Port = 8000,
  [switch]$UseMcp,
  [switch]$NoMcp,
  [switch]$Restart
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

function Get-DotEnvValue {
  param([string]$Name)
  $envPath = Join-Path (Get-Location) ".env"
  if (-not (Test-Path $envPath)) {
    return ""
  }
  $match = Get-Content $envPath | Where-Object { $_ -match "^$([regex]::Escape($Name))=" } | Select-Object -First 1
  if (-not $match) {
    return ""
  }
  return ($match -replace "^$([regex]::Escape($Name))=", "").Trim()
}

function Get-ConfigValue {
  param([string]$Name)
  $envValue = [Environment]::GetEnvironmentVariable($Name)
  if ($envValue -and $envValue.Trim()) {
    return $envValue.Trim()
  }
  return Get-DotEnvValue $Name
}

function Get-GitHubMcpToken {
  foreach ($name in @("ARC_MCP_AUTH_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "GITHUB_PERSONAL_ACCESS_TOKEN")) {
    $value = Get-ConfigValue $name
    if ($value) {
      return $value
    }
  }
  return ""
}

$env:ARC_STORAGE_PATH = ".arc/workbench-local-$Port.db"
$env:ARC_LANGGRAPH_CHECKPOINT_PATH = ".arc/langgraph-workbench-local-$Port.sqlite"
$env:ARC_STRICT_PROVIDERS = "false"
$env:ARC_JOB_QUEUE_BACKEND = "in_process"
$githubMcpToken = Get-GitHubMcpToken
$enableMcp = -not $NoMcp -and ($UseMcp -or [bool]$githubMcpToken)

if ($enableMcp) {
  $env:ARC_MCP_ENABLED = "true"
  $env:ARC_MCP_SERVER_URL = "https://api.githubcopilot.com/mcp/readonly"
  $env:ARC_MCP_TOOLS = "search_repositories,get_file_contents,search_code,list_issues,issue_read,search_issues,list_pull_requests,pull_request_read,get_latest_release"
  $env:ARC_MCP_PROMPT = "Use GitHub MCP for repository, code, issue, pull request, and release evidence; use the primary search provider for broader web context."
  $env:ARC_MCP_AUTH_REQUIRED = "true"
  if ($githubMcpToken) {
    $env:ARC_MCP_AUTH_TOKEN = $githubMcpToken
  }
} else {
  $env:ARC_MCP_ENABLED = "false"
  $env:ARC_MCP_SERVER_URL = ""
  $env:ARC_MCP_TOOLS = ""
  $env:ARC_MCP_PROMPT = ""
  $env:ARC_MCP_AUTH_REQUIRED = "false"
  $env:ARC_MCP_AUTH_TOKEN = ""
}
$env:ARC_QDRANT_URL = ""
$env:ARC_QDRANT_PREFER_LOCAL = "true"
$env:ARC_QDRANT_LOCATION = ".arc/qdrant-workbench-$Port"

$healthUrl = "http://127.0.0.1:$Port/health"
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existing -and $Restart) {
  Stop-Process -Id $existing.OwningProcess -Force
  Start-Sleep -Seconds 1
  $existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
}
if ($existing) {
  $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
  [pscustomobject]@{
    ok = $health.status -eq "ok"
    pid = $existing.OwningProcess
    url = "http://127.0.0.1:$Port/"
    health = $healthUrl
    reused = $true
    github_mcp_requested = [bool]$enableMcp
    note = "Existing process reused. Pass -Restart after changing MCP mode or environment variables."
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
for ($i = 0; $i -lt 90; $i++) {
  try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
    break
  } catch {
    Start-Sleep -Seconds 1
  }
}

if (-not $health) {
  throw "Workbench did not become healthy at $healthUrl within 90 seconds. Process id: $($process.Id)"
}

[pscustomobject]@{
  ok = $health.status -eq "ok"
  pid = $process.Id
  url = "http://127.0.0.1:$Port/"
  health = $healthUrl
  reused = $false
  github_mcp_requested = [bool]$enableMcp
} | ConvertTo-Json -Compress
