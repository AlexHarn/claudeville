#!/bin/bash
# Claudeville Startup Script
# Usage: ./start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
    echo ""
    echo "Shutting down Claudeville..."
    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "==========================================="
echo "  Claudeville"
echo "==========================================="
echo ""

# Kill any existing servers on our ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true  # Django frontend
lsof -ti:5000 | xargs kill -9 2>/dev/null || true  # Flask backend API

# Check for conda/mamba
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Using mamba for faster package management"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "Error: conda/mamba not found. Install Miniconda, Anaconda, or Mamba."
    exit 1
fi

# Check for Claude CLI
if ! command -v claude &> /dev/null; then
    echo "Warning: Claude CLI not found. Install for full functionality."
    echo ""
fi

# Create conda environment if needed
if ! conda info --envs | grep -q "claudeville"; then
    echo "Creating conda environment..."
    $CONDA_CMD env create -f environment.yaml
    echo ""
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate claudeville

# NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)" 2>/dev/null || true

# Start frontend (silently in background, logs to file)
echo "Starting frontend on http://localhost:8000 ..."
cd "$SCRIPT_DIR/environment/frontend_server"
DJANGO_LOG="$SCRIPT_DIR/django.log"
# Run migrations silently (in-memory DB, won't persist anyway)
python manage.py migrate --run-syncdb > /dev/null 2>&1 || true
# Run Django silently - all output goes to log file
python manage.py runserver > "$DJANGO_LOG" 2>&1 &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

sleep 3

if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "Error: Frontend failed to start"
    exit 1
fi

echo ""
echo "==========================================="
echo "  Frontend: http://localhost:8000"
echo "  Backend API: http://localhost:5000"
echo "  Backend CLI: Starting..."
echo "==========================================="
echo ""

# Start backend (interactive)
cd "$SCRIPT_DIR/reverie/backend_server"
python reverie.py

cleanup
