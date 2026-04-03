@echo off
REM Automated batch script for complete ML workflow
REM Usage: run_workflow.bat assam 2024-01-01 2024-03-31

SET REGION=%1
SET START_DATE=%2
SET END_DATE=%3
SET CLOUD_THRESHOLD=%4

IF "%REGION%"=="" (
    echo ERROR: Region name required
    echo Usage: run_workflow.bat assam 2024-01-01 2024-03-31 [cloud_threshold]
    exit /b 1
)

IF "%START_DATE%"=="" (
    echo ERROR: Start date required
    echo Usage: run_workflow.bat assam 2024-01-01 2024-03-31 [cloud_threshold]
    exit /b 1
)

IF "%END_DATE%"=="" (
    echo ERROR: End date required
    echo Usage: run_workflow.bat assam 2024-01-01 2024-03-31 [cloud_threshold]
    exit /b 1
)

echo ================================================================================
echo ML WORKFLOW - BATCH EXECUTION
echo ================================================================================
echo Region: %REGION%
echo Date Range: %START_DATE% to %END_DATE%
IF NOT "%CLOUD_THRESHOLD%"=="" echo Cloud Threshold: %CLOUD_THRESHOLD%%%
echo ================================================================================
echo.

REM Activate virtual environment if it exists
IF EXIST "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    CALL venv\Scripts\activate.bat
)

REM Run workflow
python run_full_workflow.py ^
    --region %REGION% ^
    --start-date %START_DATE% ^
    --end-date %END_DATE% ^
    --cloud-threshold %CLOUD_THRESHOLD% ^
    --tile-size 128 ^
    --epochs 400

echo.
echo Workflow completed!
pause
