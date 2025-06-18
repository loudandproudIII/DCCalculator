param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Packages)

if ($Packages.Count -eq 0) {
    Write-Host "Usage: .\install.ps1 package_name"
    exit
}

Write-Host "Installing: $($Packages -join ' ')"
$VenvDirs = Get-ChildItem -Directory -Name "venv_*"

foreach ($VenvName in $VenvDirs) {
    Write-Host "Installing to $VenvName..." -ForegroundColor Yellow
    
    # Activate the venv and install normally
    & "$VenvName\Scripts\Activate.ps1"
    pip install @Packages --quiet
    deactivate
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "$VenvName - SUCCESS" -ForegroundColor Green
    } else {
        Write-Host "$VenvName - FAILED" -ForegroundColor Red
    }
}

Write-Host "Done!" -ForegroundColor Green