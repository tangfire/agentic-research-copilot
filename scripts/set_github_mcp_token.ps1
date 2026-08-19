param(
  [string]$EnvPath = ".env"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

$resolvedEnvPath = Join-Path $repoRoot $EnvPath
$secureToken = Read-Host "Paste a NEW GitHub token for ARC_MCP_AUTH_TOKEN" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
try {
  $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
} finally {
  if ($bstr -ne [IntPtr]::Zero) {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
  }
}

if (-not $token -or -not $token.Trim()) {
  throw "Token is empty. No changes were written."
}

$lines = @()
if (Test-Path $resolvedEnvPath) {
  $lines = @(Get-Content $resolvedEnvPath)
}

$updated = $false
$nextLines = foreach ($line in $lines) {
  if ($line -match '^ARC_MCP_AUTH_TOKEN=') {
    $updated = $true
    "ARC_MCP_AUTH_TOKEN=$token"
  } else {
    $line
  }
}

if (-not $updated) {
  $nextLines += "ARC_MCP_AUTH_TOKEN=$token"
}

Set-Content -Path $resolvedEnvPath -Value $nextLines -Encoding utf8

[pscustomobject]@{
  ok = $true
  env_path = $resolvedEnvPath
  token_configured = $true
  note = "Token written to local .env. Do not commit .env."
} | ConvertTo-Json -Compress
