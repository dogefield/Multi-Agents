# Install MCP Servers for Claude Desktop
Write-Host "Installing MCP Servers for Enhanced Claude Capabilities..." -ForegroundColor Green

# Check if npm is installed
try {
    npm --version | Out-Null
    Write-Host "✓ npm is installed" -ForegroundColor Green
} catch {
    Write-Host "✗ npm is not installed. Please install Node.js first." -ForegroundColor Red
    Write-Host "Download from: https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

Write-Host "`nInstalling MCP servers..." -ForegroundColor Yellow

# Git MCP Server (for Git operations)
Write-Host "`n1. Installing Git MCP Server..." -ForegroundColor Cyan
npm install -g @modelcontextprotocol/server-git

# PostgreSQL MCP Server (for direct database queries)
Write-Host "`n2. Installing PostgreSQL MCP Server..." -ForegroundColor Cyan
npm install -g @modelcontextprotocol/server-postgres

# Puppeteer MCP Server (for browser automation and dev tools)
Write-Host "`n3. Installing Puppeteer MCP Server..." -ForegroundColor Cyan
npm install -g @modelcontextprotocol/server-puppeteer

# Logs MCP Server (for real-time log monitoring)
Write-Host "`n4. Installing Logs MCP Server..." -ForegroundColor Cyan
npm install -g @modelcontextprotocol/server-logs

# Network MCP Server (for network diagnostics)
Write-Host "`n5. Installing Network MCP Server..." -ForegroundColor Cyan
npm install -g @modelcontextprotocol/server-network

Write-Host "`n✓ All MCP servers installed!" -ForegroundColor Green

# Backup current config
Write-Host "`nBacking up current Claude config..." -ForegroundColor Yellow
$configPath = "$env:APPDATA\Claude\claude_desktop_config.json"
$backupPath = "$env:APPDATA\Claude\claude_desktop_config_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"

if (Test-Path $configPath) {
    Copy-Item $configPath $backupPath
    Write-Host "✓ Config backed up to: $backupPath" -ForegroundColor Green
}

# Apply new config
Write-Host "`nApplying new configuration..." -ForegroundColor Yellow
$newConfigPath = "$env:APPDATA\Claude\claude_desktop_config_new.json"
if (Test-Path $newConfigPath) {
    Copy-Item $newConfigPath $configPath -Force
    Write-Host "✓ New configuration applied!" -ForegroundColor Green
    Remove-Item $newConfigPath
} else {
    Write-Host "✗ New config file not found!" -ForegroundColor Red
}

Write-Host "`n⚠️  IMPORTANT: You need to restart Claude Desktop for the changes to take effect!" -ForegroundColor Yellow
Write-Host "After restarting, you'll have access to:" -ForegroundColor Cyan
Write-Host "  - Git operations (commit, push, pull)" -ForegroundColor White
Write-Host "  - Direct database queries to PostgreSQL/Supabase" -ForegroundColor White
Write-Host "  - Browser automation with dev tools access" -ForegroundColor White
Write-Host "  - Real-time log file monitoring" -ForegroundColor White
Write-Host "  - Network diagnostics (ping, port checks)" -ForegroundColor White
Write-Host "  - Admin operations (with elevation)" -ForegroundColor White

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
