<#
  Run the dashboard in public mode on this PC and open an HTTPS tunnel to it.

    powershell -ExecutionPolicy Bypass -File scripts\run_public.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_public.ps1 -Tunnel ssh

  Needs an admin account first:  python -m dashboard.app create-user --role admin
  The public address is printed by the tunnel and changes every time you start it.
  Press Ctrl+C to stop both the tunnel and the server.
#>
param(
    [int]$Port = 5000,
    [ValidateSet("cloudflared", "ssh")] [string]$Tunnel = "cloudflared"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root "..\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# Find the tunnel program before starting anything.
if ($Tunnel -eq "cloudflared") {
    $cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
    if (-not $cf -and (Test-Path "$root\tools\cloudflared.exe")) { $cf = "$root\tools\cloudflared.exe" }
    if (-not $cf) { throw "cloudflared not found. Put cloudflared.exe on PATH or in $root\tools, or run with -Tunnel ssh." }
}

Write-Host "Starting the dashboard (public mode) on 127.0.0.1:$Port ..."
$server = Start-Process -FilePath $py -ArgumentList @("-m", "dashboard.app", "serve", "--public", "--port", $Port) `
    -PassThru -NoNewWindow
try {
    Start-Sleep -Seconds 4
    if ($server.HasExited) { throw "The server did not start (see the message above)." }
    Write-Host "Opening the tunnel. Share the https:// address it prints. Ctrl+C stops everything."
    if ($Tunnel -eq "cloudflared") {
        & $cf tunnel --url "http://127.0.0.1:$Port"
    } else {
        & ssh -o ServerAliveInterval=30 -R "80:127.0.0.1:$Port" nokey@localhost.run
    }
} finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    Write-Host "Stopped."
}
