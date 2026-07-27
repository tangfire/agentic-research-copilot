param(
  [int]$Port = 8000,
  [switch]$Reload,
  [switch]$Smoke,
  [switch]$InProcess,
  [switch]$NoWorker,
  [switch]$NoMcp,
  [switch]$NoInfra
)

$ErrorActionPreference = "Stop"

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

function Test-TcpPort {
  param(
    [string]$HostName,
    [int]$PortNumber
  )
  $result = Test-NetConnection -ComputerName $HostName -Port $PortNumber -WarningAction SilentlyContinue
  return [bool]$result.TcpTestSucceeded
}

function Assert-PythonModule {
  param([string]$ModuleName)
  python -c "import $ModuleName" | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "Python module '$ModuleName' is missing. Run: pip install -e .[queue]"
  }
}

function Start-DockerService {
  param(
    [string]$Name,
    [string]$Image,
    [string[]]$DockerArgs
  )
  $docker = Get-Command docker -ErrorAction SilentlyContinue
  if (-not $docker) {
    throw "Docker is required to auto-start $Name, or start the service manually."
  }
  $existing = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $Name } | Select-Object -First 1
  if ($existing) {
    docker start $Name | Out-Null
    return
  }
  docker run -d --name $Name @DockerArgs $Image | Out-Null
}

function Ensure-Redis {
  if (Test-TcpPort -HostName "localhost" -PortNumber 6379) {
    return
  }
  if ($NoInfra) {
    throw "Redis is not reachable on localhost:6379."
  }
  Start-DockerService -Name "arc-redis" -Image "redis:7-alpine" -DockerArgs @("-p", "6379:6379", "-v", "arc-redis-data:/data")
  for ($i = 0; $i -lt 20; $i += 1) {
    if (Test-TcpPort -HostName "localhost" -PortNumber 6379) {
      return
    }
    Start-Sleep -Seconds 1
  }
  throw "Redis did not become ready on localhost:6379."
}

function Ensure-Qdrant {
  if (Test-TcpPort -HostName "localhost" -PortNumber 6333) {
    return
  }
  if ($NoInfra) {
    throw "Qdrant is not reachable on localhost:6333."
  }
  Start-DockerService `
    -Name "arc-qdrant" `
    -Image "ghcr.io/qdrant/qdrant/qdrant:v1.18.0" `
    -DockerArgs @("-p", "6333:6333", "-v", "arc-qdrant-data:/qdrant/storage")
  for ($i = 0; $i -lt 30; $i += 1) {
    if (Test-TcpPort -HostName "localhost" -PortNumber 6333) {
      return
    }
    Start-Sleep -Seconds 1
  }
  throw "Qdrant did not become ready on localhost:6333."
}

function Start-CeleryWorker {
  Assert-PythonModule -ModuleName "celery"
  Assert-PythonModule -ModuleName "redis"
  $existing = Get-CimInstance Win32_Process |
    Where-Object {
      $_.CommandLine -match "agentic_research_copilot\.celery_app" -and
      $_.CommandLine -match "worker"
    } |
    Select-Object -First 1
  if ($existing) {
    Write-Host "Celery worker already running with PID $($existing.ProcessId)."
    return
  }
  $logDir = Join-Path (Get-Location) ".arc"
  New-Item -ItemType Directory -Force -Path $logDir | Out-Null
  $stdout = Join-Path $logDir "celery-worker.out.log"
  $stderr = Join-Path $logDir "celery-worker.err.log"
  $args = @(
    "-m",
    "celery",
    "-A",
    "agentic_research_copilot.celery_app",
    "worker",
    "--loglevel=INFO",
    "--pool=solo",
    "--concurrency=1"
  )
  $process = Start-Process `
    -FilePath "python" `
    -ArgumentList $args `
    -WorkingDirectory (Get-Location) `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru
  Start-Sleep -Seconds 3
  if ($process.HasExited) {
    throw "Celery worker exited early. Check $stderr"
  }
  Write-Host "Celery worker started with PID $($process.Id). Logs: $stdout"
}

function Start-McpServer {
  if ($NoMcp) {
    $env:ARC_MCP_ENABLED = "false"
    Write-Host "MCP workbench server disabled by -NoMcp."
    return
  }
  if ($env:ARC_MCP_ENABLED -match "^(0|false|no|off)$") {
    Write-Host "MCP workbench server disabled by ARC_MCP_ENABLED=$env:ARC_MCP_ENABLED."
    return
  }
  Assert-PythonModule -ModuleName "mcp.server.fastmcp"
  if (-not $env:ARC_MCP_SERVER_URL) {
    $env:ARC_MCP_SERVER_URL = "http://127.0.0.1:8765"
  }
  if (-not $env:ARC_MCP_TOOLS) {
    $env:ARC_MCP_TOOLS = "search_grounding_corpus,recall_project_memory,inspect_research_runs,check_demo_readiness"
  }
  if (-not $env:ARC_MCP_PROMPT) {
    $env:ARC_MCP_PROMPT = "Use MCP workspace tools when ingested grounding documents, project memory, prior run traces/evaluation, or demo readiness checks can improve the research answer."
  }
  if (-not $env:ARC_MCP_DEMO_API_BASE) {
    $env:ARC_MCP_DEMO_API_BASE = "http://127.0.0.1:$Port"
  }

  $uri = [uri]$env:ARC_MCP_SERVER_URL
  $hostName = if ($uri.Host) { $uri.Host } else { "127.0.0.1" }
  $portNumber = if ($uri.Port -gt 0) { $uri.Port } elseif ($uri.Scheme -eq "https") { 443 } else { 80 }
  if (-not $env:ARC_MCP_DEMO_PORT) {
    $env:ARC_MCP_DEMO_PORT = "$portNumber"
  }
  if (Test-TcpPort -HostName $hostName -PortNumber $portNumber) {
    Write-Host "MCP workbench server already reachable at $env:ARC_MCP_SERVER_URL."
    return
  }

  $logDir = Join-Path (Get-Location) ".arc"
  New-Item -ItemType Directory -Force -Path $logDir | Out-Null
  $stdout = Join-Path $logDir "mcp-server.out.log"
  $stderr = Join-Path $logDir "mcp-server.err.log"
  $process = Start-Process `
    -FilePath "python" `
    -ArgumentList @("-m", "agentic_research_copilot.research_mcp_server") `
    -WorkingDirectory (Get-Location) `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

  for ($i = 0; $i -lt 15; $i += 1) {
    if ($process.HasExited) {
      throw "MCP workbench server exited early. Check $stderr"
    }
    if (Test-TcpPort -HostName $hostName -PortNumber $portNumber) {
      Write-Host "MCP workbench server started with PID $($process.Id). URL: $env:ARC_MCP_SERVER_URL Logs: $stdout"
      return
    }
    Start-Sleep -Seconds 1
  }
  throw "MCP workbench server did not become ready on $env:ARC_MCP_SERVER_URL. Check $stderr"
}

$env:ARC_STRICT_PROVIDERS = "true"

if (-not $env:ARC_MODEL_PROVIDER) {
  $env:ARC_MODEL_PROVIDER = "openai_compatible"
}
if (-not $env:ARC_EMBEDDING_PROVIDER) {
  $env:ARC_EMBEDDING_PROVIDER = "openai_compatible"
}
if (-not $env:ARC_EMBEDDING_BASE_URL) {
  $env:ARC_EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
}
if (-not $env:ARC_SEARCH_PROVIDER) {
  $env:ARC_SEARCH_PROVIDER = "tavily"
}
if (-not $env:ARC_SEARCH_DEPTH) {
  $env:ARC_SEARCH_DEPTH = "basic"
}
if (-not $env:ARC_SEARCH_INCLUDE_RAW_CONTENT) {
  $env:ARC_SEARCH_INCLUDE_RAW_CONTENT = "true"
}
if (-not $env:ARC_MCP_ENABLED) {
  $env:ARC_MCP_ENABLED = "true"
}
if (-not $env:ARC_MCP_SERVER_URL) {
  $env:ARC_MCP_SERVER_URL = "http://127.0.0.1:8765"
}
if (-not $env:ARC_MCP_TOOLS) {
  $env:ARC_MCP_TOOLS = "search_grounding_corpus,recall_project_memory,inspect_research_runs,check_demo_readiness"
}
if (-not $env:ARC_MCP_PROMPT) {
  $env:ARC_MCP_PROMPT = "Use MCP workspace tools when ingested grounding documents, project memory, prior run traces/evaluation, or demo readiness checks can improve the research answer."
}
if (-not $env:ARC_MCP_TRANSPORT) {
  $env:ARC_MCP_TRANSPORT = "streamable_http"
}
if (-not $env:ARC_SOURCE_READER_ENABLED) {
  $env:ARC_SOURCE_READER_ENABLED = "true"
}
if (-not $env:ARC_SOURCE_READER_STRATEGY) {
  $env:ARC_SOURCE_READER_STRATEGY = "chunk_rerank_compress"
}
if (-not $env:ARC_SOURCE_READER_MAX_CHARS) {
  $env:ARC_SOURCE_READER_MAX_CHARS = "50000"
}
if (-not $env:ARC_SOURCE_READER_EXCERPT_CHARS) {
  $env:ARC_SOURCE_READER_EXCERPT_CHARS = "1600"
}
if (-not $env:ARC_SOURCE_READER_CHUNK_CONTEXT_WINDOW) {
  $env:ARC_SOURCE_READER_CHUNK_CONTEXT_WINDOW = "1"
}
if (-not $env:ARC_RERANK_PROVIDER) {
  $env:ARC_RERANK_PROVIDER = "dashscope"
}
if (-not $env:ARC_RERANK_BASE_URL) {
  $env:ARC_RERANK_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
}
if (-not $env:ARC_RERANK_MODEL) {
  $env:ARC_RERANK_MODEL = "qwen3-rerank"
}

if ($InProcess) {
  $env:ARC_JOB_QUEUE_BACKEND = "in_process"
} else {
  $env:ARC_JOB_QUEUE_BACKEND = "celery"
  if (-not $env:ARC_CELERY_BROKER_URL) {
    $env:ARC_CELERY_BROKER_URL = "redis://localhost:6379/0"
  }
  if (-not $env:ARC_CELERY_RESULT_BACKEND) {
    $env:ARC_CELERY_RESULT_BACKEND = "redis://localhost:6379/1"
  }
  if (-not $env:ARC_QDRANT_URL) {
    $env:ARC_QDRANT_URL = "http://localhost:6333"
  }
  $env:ARC_QDRANT_PREFER_LOCAL = "false"
  Ensure-Redis
  Ensure-Qdrant
}

$dotenvQdrantUrl = Get-DotEnvValue "ARC_QDRANT_URL"
$dotenvQdrantLocation = Get-DotEnvValue "ARC_QDRANT_LOCATION"
if (
  -not $env:ARC_QDRANT_URL `
  -and -not $dotenvQdrantUrl `
  -and (-not $env:ARC_QDRANT_LOCATION -or $env:ARC_QDRANT_LOCATION -eq ":memory:") `
  -and (-not $dotenvQdrantLocation -or $dotenvQdrantLocation -eq ":memory:")
) {
  $env:ARC_QDRANT_URL = "http://localhost:6333"
}
if (-not $env:ARC_LANGGRAPH_CHECKPOINTER) {
  $env:ARC_LANGGRAPH_CHECKPOINTER = "sqlite"
}

$checkArgs = @("scripts/check_real_providers.py")
if ($Smoke) {
  $checkArgs += "--smoke"
}
python @checkArgs
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

if ($env:ARC_JOB_QUEUE_BACKEND -eq "celery" -and -not $NoWorker) {
  Start-CeleryWorker
}

Start-McpServer

$uvicornArgs = @(
  "-m",
  "uvicorn",
  "agentic_research_copilot.server:app",
  "--host",
  "127.0.0.1",
  "--port",
  "$Port"
)
if ($Reload) {
  $uvicornArgs += "--reload"
}

python @uvicornArgs
