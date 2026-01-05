#!/bin/bash
set -e  # Exit on error

ENV_FILE=".env"
REMOTE_NAME="my_remote"

# 1. Load & Verify Secrets
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: $ENV_FILE not found."
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [ -z "$GDRIVE_CLIENT_ID" ] || [ -z "$GDRIVE_CLIENT_SECRET" ]; then
    echo "❌ Error: Missing credentials in $ENV_FILE."
    exit 1
fi

# 2. Setup Environment (uv)
echo "🐍 Setting up Python environment..."

# Create venv if missing
if [ ! -d ".venv" ]; then
    uv venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install "dvc[gdrive]"

# 3. Configure DVC & Pull
echo "🔐 Configuring DVC and pulling data..."

dvc remote modify --local "$REMOTE_NAME" gdrive_client_id "$GDRIVE_CLIENT_ID"
dvc remote modify --local "$REMOTE_NAME" gdrive_client_secret "$GDRIVE_CLIENT_SECRET"

dvc pull

echo "✅ Setup Complete."