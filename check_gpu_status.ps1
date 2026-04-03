# HP Pavilion GPU Enablement Script for Windows 11
# Run this in PowerShell (as Administrator recommended)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "HP PAVILION - NVIDIA GPU ENABLEMENT CHECK" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check 1: Verify NVIDIA GPU is detected by Windows
Write-Host "[1/5] Checking for NVIDIA GPU..." -ForegroundColor Yellow
$gpu = Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"}
if ($gpu) {
    Write-Host "  ✓ FOUND: $($gpu.Name)" -ForegroundColor Green
    Write-Host "  Status: $($gpu.Status)" -ForegroundColor Gray
} else {
    Write-Host "  ✗ NVIDIA GPU not found!" -ForegroundColor Red
    Write-Host "  This could mean:" -ForegroundColor Yellow
    Write-Host "    - GPU is disabled in BIOS" -ForegroundColor Yellow
    Write-Host "    - Hardware issue" -ForegroundColor Yellow
    Write-Host "    - Driver not installed" -ForegroundColor Yellow
}

Write-Host ""

# Check 2: Display all video controllers
Write-Host "[2/5] All Video Controllers:" -ForegroundColor Yellow
Get-WmiObject Win32_VideoController | ForEach-Object {
    Write-Host "  • $($_.Name)" -ForegroundColor Gray
    Write-Host "    Status: $($_.Status)" -ForegroundColor Gray
}

Write-Host ""

# Check 3: Check PyTorch CUDA availability
Write-Host "[3/5] Testing PyTorch CUDA..." -ForegroundColor Yellow
try {
    & python -c "import torch; print('CUDA:', torch.cuda.is_available())"
    Write-Host "  Run the command above manually to see CUDA status" -ForegroundColor Gray
} catch {
    Write-Host "  ✗ Error testing PyTorch: $_" -ForegroundColor Red
}

Write-Host ""

# Check 4: Check graphics settings registry
Write-Host "[4/5] Checking Graphics Settings..." -ForegroundColor Yellow
try {
    $graphicsPath = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
    if (Test-Path $graphicsPath) {
        Write-Host "  ✓ Graphics drivers registry accessible" -ForegroundColor Green
        
        # Check for hardware acceleration setting
        $hagEnabled = Get-ItemProperty -Path $graphicsPath -Name "HwSchMode" -ErrorAction SilentlyContinue
        if ($hagEnabled.HwSchMode -eq 2) {
            Write-Host "  ✓ Hardware-accelerated GPU scheduling: ENABLED" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ Hardware-accelerated GPU scheduling: DISABLED" -ForegroundColor Yellow
            Write-Host "    To enable: Settings → System → Display → Graphics → Change default graphics settings" -ForegroundColor Gray
        }
    } else {
        Write-Host "  ✗ Cannot access graphics registry" -ForegroundColor Red
    }
} catch {
    Write-Host "  ✗ Error checking registry: $_" -ForegroundColor Red
    Write-Host "    Try running as Administrator" -ForegroundColor Yellow
}

Write-Host ""

# Check 5: Check if NVIDIA services are running
Write-Host "[5/5] Checking NVIDIA Services..." -ForegroundColor Yellow
$nvidiaServices = @("nvlddmkm", "NVDisplay.ContainerLocalSystem", "NvTelemetryContainer")
foreach ($service in $nvidiaServices) {
    $svc = Get-Service -Name $service -ErrorAction SilentlyContinue
    if ($svc) {
        if ($svc.Status -eq "Running") {
            Write-Host "  ✓ $service : Running" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ $service : $($svc.Status) (Should be Running)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ✗ $service : Not found" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DIAGNOSIS COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Provide recommendations
Write-Host "RECOMMENDATIONS:" -ForegroundColor Cyan
Write-Host ""

if (-not $gpu) {
    Write-Host "❌ NVIDIA GPU not detected by Windows" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try these steps IN ORDER:" -ForegroundColor Yellow
    Write-Host "  1. Restart your laptop (sometimes re-enables GPU)" -ForegroundColor White
    Write-Host "  2. Update/reinstall NVIDIA drivers from nvidia.com" -ForegroundColor White
    Write-Host "  3. Check BIOS settings (F10 on boot) for 'Discrete GPU' or 'Switchable Graphics'" -ForegroundColor White
    Write-Host "  4. Contact HP Support if under warranty" -ForegroundColor White
    Write-Host ""
    Write-Host "Alternative: Use Google Colab (free GPU cloud)" -ForegroundColor Green
    Write-Host "  Upload: MAE_Training_Colab.ipynb" -ForegroundColor Gray
} else {
    Write-Host "✓ NVIDIA GPU detected by Windows" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps to enable in PyTorch:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "STEP 1: Windows Graphics Settings" -ForegroundColor Cyan
    Write-Host "  1. Press Win+I → System → Display → Graphics" -ForegroundColor White
    Write-Host "  2. Turn ON 'Hardware-accelerated GPU scheduling'" -ForegroundColor White
    Write-Host "  3. Add python.exe and set to 'High Performance'" -ForegroundColor White
    Write-Host "  4. Restart laptop" -ForegroundColor White
    Write-Host ""
    Write-Host "STEP 2: NVIDIA Control Panel" -ForegroundColor Cyan
    Write-Host "  1. Right-click desktop → NVIDIA Control Panel" -ForegroundColor White
    Write-Host "  2. Manage 3D Settings → Program Settings" -ForegroundColor White
    Write-Host "  3. Add python.exe" -ForegroundColor White
    Write-Host "  4. Select 'High-performance NVIDIA processor'" -ForegroundColor White
    Write-Host "  5. Click Apply" -ForegroundColor White
    Write-Host ""
    Write-Host "STEP 3: Test Again" -ForegroundColor Cyan
    Write-Host "  Run: python check_cuda.py" -ForegroundColor White
    Write-Host ""
    Write-Host "If still failing after restart, use Google Colab instead" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
