#!/bin/bash
# Automated workflow script for Linux/Mac
# Usage: ./run_workflow.sh assam 2024-01-01 2024-03-31

REGION=$1
START_DATE=$2
END_DATE=$3
CLOUD_THRESHOLD=${4:-}

if [ -z "$REGION" ]; then
    echo "ERROR: Region name required"
    echo "Usage: ./run_workflow.sh assam 2024-01-01 2024-03-31 [cloud_threshold]"
    exit 1
fi

if [ -z "$START_DATE" ]; then
    echo "ERROR: Start date required"
    echo "Usage: ./run_workflow.sh assam 2024-01-01 2024-03-31 [cloud_threshold]"
    exit 1
fi

if [ -z "$END_DATE" ]; then
    echo "ERROR: End date required"
    echo "Usage: ./run_workflow.sh assam 2024-01-01 2024-03-31 [cloud_threshold]"
    exit 1
fi

echo "================================================================================"
echo "ML WORKFLOW - AUTOMATED EXECUTION"
echo "================================================================================"
echo "Region: $REGION"
echo "Date Range: $START_DATE to $END_DATE"
[ ! -z "$CLOUD_THRESHOLD" ] && echo "Cloud Threshold: ${CLOUD_THRESHOLD}%"
echo "================================================================================"
echo ""

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run workflow
python run_full_workflow.py \
    --region "$REGION" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --cloud-threshold "$CLOUD_THRESHOLD" \
    --tile-size 128 \
    --epochs 400

echo ""
echo "Workflow completed!"
