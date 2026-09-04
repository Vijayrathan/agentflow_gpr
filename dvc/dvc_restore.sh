#!/bin/bash
set -e 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../backend/.env"
REMOTE_NAME="my_remote"

# 1. Load & Verify Secrets
if [ ! -f "$ENV_FILE" ]; then
    echo " Error: $ENV_FILE not found."
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [ -z "$GDRIVE_CLIENT_ID" ] || [ -z "$GDRIVE_CLIENT_SECRET" ]; then
    echo "Error: Missing credentials in $ENV_FILE."
    exit 1
fi


# 3. Configure DVC & Pull
echo "Configuring DVC and pulling data..."

dvc remote modify --local "$REMOTE_NAME" gdrive_client_id "$GDRIVE_CLIENT_ID"
dvc remote modify --local "$REMOTE_NAME" gdrive_client_secret "$GDRIVE_CLIENT_SECRET"

dvc pull

echo "Setup Complete."