#!/bin/bash
# Setup TGP Environment
#
# Usage: ./run/setup_environment.sh

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}TGP Environment Setup${NC}"
echo -e "${GREEN}================================================${NC}"

# Activate conda environment
echo -e "\n${YELLOW}[1/5] Activating conda environment...${NC}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llms
echo "Python: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Install dependencies
echo -e "\n${YELLOW}[2/5] Installing dependencies...${NC}"
pip install -q redis sentence-transformers python-dotenv anthropic codecarbon

# Start Redis
echo -e "\n${YELLOW}[3/5] Starting Redis server...${NC}"
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "Redis already running"
    else
        redis-server --daemonize yes --port 6379
        sleep 1
        if redis-cli ping &> /dev/null; then
            echo -e "${GREEN}Redis started successfully${NC}"
        else
            echo -e "${RED}Failed to start Redis${NC}"
        fi
    fi
else
    echo "Installing Redis..."
    conda install -c conda-forge redis -y
    redis-server --daemonize yes --port 6379
fi

# Create directories
echo -e "\n${YELLOW}[4/5] Creating directories...${NC}"
mkdir -p data/{raw,processed,training}
mkdir -p output/{models,results,figures,tables}
mkdir -p paper/sections
mkdir -p tests
echo "Directories created"

# Verify setup
echo -e "\n${YELLOW}[5/5] Verifying setup...${NC}"
python scripts/verify_setup.py

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Generate training data: python scripts/generate_training_data.py"
echo "  2. Run quick test: python scripts/quick_test.py"
echo "  3. Full pipeline: ./run/run_all.sh"
