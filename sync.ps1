param(
    [switch]$Force,      # Force reinstall all packages
    [switch]$DryRun      # Show what would be done without doing it
)

if ($DryRun) {
    Write-Host "DRY RUN MODE - No changes will be made" -ForegroundColor Magenta
    Write-Host ("=" * 50)
}

Write-Host "Synchronizing all environments with requirements.txt..." -ForegroundColor Green
if ($Force) { Write-Host "FORCE MODE: Reinstalling all packages" -ForegroundColor Yellow }

# Check if requirements.txt exists
if (-not (Test-Path "requirements.txt")) {
    Write-Host "requirements.txt not found!" -ForegroundColor Red
    Write-Host "Please run install.ps1 to install some packages first." -ForegroundColor Yellow
    exit
}

# Find all venv directories
$VenvDirs = Get-ChildItem -Directory -Name "venv_*"

if ($VenvDirs.Count -eq 0) {
    Write-Host "No virtual environments found!" -ForegroundColor Red
    Write-Host "Please run setup.ps1 first to create an environment." -ForegroundColor Yellow
    exit
}

Write-Host "Found $($VenvDirs.Count) environment(s) to sync" -ForegroundColor Cyan
Write-Host ""

# Show what we're syncing to
Write-Host "Requirements to sync:" -ForegroundColor Cyan
$RequirementsContent = Get-Content "requirements.txt"
$PackageCount = ($RequirementsContent | Where-Object { $_ -match "^[a-zA-Z]" }).Count
Write-Host "  Total packages: $PackageCount" -ForegroundColor Gray
$RequirementsContent | Where-Object { $_ -match "^[a-zA-Z]" } | Select-Object -First 3 | ForEach-Object { 
    $PackageName = ($_ -split "==")[0]
    Write-Host "  $PackageName" -ForegroundColor Gray 
}
if ($PackageCount -gt 3) {
    Write-Host "  ... and $($PackageCount - 3) more packages" -ForegroundColor Gray
}

if ($DryRun) {
    Write-Host ""
    Write-Host "Would sync these $PackageCount packages to all $($VenvDirs.Count) environments." -ForegroundColor Magenta
    Write-Host "Run without -DryRun to perform actual sync." -ForegroundColor Magenta
    exit
}

Write-Host ""

$SyncSuccess = 0
$SyncFailed = 0

# Sync each environment using pip install --target (same approach as install.ps1)
foreach ($VenvName in $VenvDirs) {
    $TargetPath = Join-Path $VenvName "Lib\site-packages"
    
    if (-not (Test-Path $TargetPath)) {
        Write-Host "[$VenvName] - Invalid environment, skipping" -ForegroundColor Red
        $SyncFailed++
        continue
    }
    
    Write-Host "[$VenvName] Syncing..." -ForegroundColor Yellow
    
    try {
        # Clean up __pycache__ directories first to avoid permission issues
        Get-ChildItem -Path $TargetPath -Recurse -Directory -Name "__pycache__" -ErrorAction SilentlyContinue | ForEach-Object {
            $PycachePath = Join-Path $TargetPath $_
            Remove-Item $PycachePath -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        # Build pip install command arguments
        $InstallArgs = @(
            "--target", $TargetPath
            "--no-deps"  # Don't install dependencies to avoid conflicts
            "-r", "requirements.txt"
            "--quiet"
        )
        
        if ($Force) {
            $InstallArgs += "--force-reinstall"
        }
        
        # Use YOUR pip with --target (same reliable approach as install.ps1)
        pip install @InstallArgs
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$VenvName] - SUCCESS" -ForegroundColor Green
            $SyncSuccess++
        } else {
            Write-Host "[$VenvName] - FAILED" -ForegroundColor Red
            $SyncFailed++
        }
    }
    catch {
        Write-Host "[$VenvName] - ERROR: $($_.Exception.Message)" -ForegroundColor Red
        $SyncFailed++
    }
}

Write-Host ""
Write-Host ("=" * 50)
Write-Host "SYNC SUMMARY:" -ForegroundColor White
Write-Host "  Environments processed: $($VenvDirs.Count)" -ForegroundColor White
Write-Host "  Successful syncs: $SyncSuccess" -ForegroundColor Green
Write-Host "  Failed syncs: $SyncFailed" -ForegroundColor Red

if ($SyncFailed -gt 0) {
    Write-Host ""
    Write-Host "Some synchronizations failed. Try:" -ForegroundColor Yellow
    Write-Host "  .\sync.ps1 -Force    # Force reinstall" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "🎉 All environments are synchronized!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Available commands:" -ForegroundColor White
Write-Host "  .\install.ps1 package_name  - Install new packages" -ForegroundColor Gray
Write-Host "  .\sync.ps1 -Force           - Force reinstall all packages" -ForegroundColor Gray
Write-Host "  .\run.ps1                   - Run your code" -ForegroundColor Gray