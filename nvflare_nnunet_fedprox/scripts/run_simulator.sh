#!/bin/bash
###############################################################################
# NVIDIA FLARE + nnU-Net V2 + FedProx - Simulator Mode Launcher
#
# This script uses NVFLARE Simulator (like your previous successful project)
# instead of POC mode for better stability and simpler deployment.
#
# Usage:
#   bash run_simulator.sh
###############################################################################

set -e

# ============================================================================
# Environment Setup
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# nnU-Net environment variables
if [ -z "$nnUNet_raw" ] || [ -z "$nnUNet_preprocessed" ] || [ -z "$nnUNet_results" ]; then
    echo "✗ Error: nnU-Net environment variables not set"
    exit 1
fi

echo "nnUNet environment:"
echo "  nnUNet_raw: $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results: $nnUNet_results"
echo ""

# Workspace paths
FL_JOB_DIR="$PROJECT_ROOT/fl_job"
WORKSPACE_DIR="/root/autodl-tmp/workspace_poc"

# ============================================================================
# Create Job Directory Structure
# ============================================================================
echo "========================================="
echo "Creating FL Job Structure..."
echo "========================================="

# Clean old job
rm -rf "$FL_JOB_DIR"
mkdir -p "$FL_JOB_DIR/app/config"
mkdir -p "$FL_JOB_DIR/app/custom"

# Copy custom code
echo "Copying custom code..."
cp -r "$PROJECT_ROOT/custom/"* "$FL_JOB_DIR/app/custom/"
echo "✓ Custom code copied"

# Create clean configurations (remove comment fields)
echo "Creating clean configurations..."

# Clean server config
python3 << 'PYEOF'
import json
import sys

# Read original config
with open('/root/autodl-tmp/nvflare_nnunet_fedprox/config/config_fed_server.json', 'r') as f:
    config = json.load(f)

# Function to remove comment keys and problematic fields
def clean_dict(d):
    if isinstance(d, dict):
        cleaned = {}
        for k, v in d.items():
            # Skip comment keys, model definition, and save_name (not accepted by class)
            if k.startswith('###') or k == 'model' or k == 'save_name':
                continue
            cleaned[k] = clean_dict(v)
        return cleaned
    elif isinstance(d, list):
        return [clean_dict(item) for item in d]
    else:
        return d

cleaned = clean_dict(config)

# Write cleaned config
with open('/root/autodl-tmp/nvflare_nnunet_fedprox/fl_job/app/config/config_fed_server.json', 'w') as f:
    json.dump(cleaned, f, indent=2)

print("✓ Server config cleaned")
PYEOF

# Clean client config (create generic config)
python3 << 'PYEOF'
import json
import sys

# Function to remove comment keys
def clean_dict(d):
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items() if not k.startswith('###')}
    elif isinstance(d, list):
        return [clean_dict(item) for item in d]
    else:
        return d

# Use site-1 config as base template (dataset will be determined by executor based on site name)
with open('/root/autodl-tmp/nvflare_nnunet_fedprox/config/config_fed_client_site1.json', 'r') as f:
    config = json.load(f)

cleaned = clean_dict(config)

# Set dataset to a placeholder - executor will override based on site name
cleaned['executors'][0]['executor']['args']['dataset_name_or_id'] = 'PLACEHOLDER_WILL_BE_DETERMINED_BY_SITE'

# Write generic client config
with open('/root/autodl-tmp/nvflare_nnunet_fedprox/fl_job/app/config/config_fed_client.json', 'w') as f:
    json.dump(cleaned, f, indent=2)

print("✓ Generic client config created")
PYEOF

echo "✓ Configurations cleaned and copied"

# Create meta.json
echo "Creating meta.json..."
cat > "$FL_JOB_DIR/meta.json" << 'EOF'
{
  "name": "nnunet_fedprox_fl",
  "resource_spec": {},
  "deploy_map": {
    "app": [
      "@ALL"
    ]
  },
  "min_clients": 2,
  "mandatory_clients": []
}
EOF
echo "✓ meta.json created"

echo ""
echo "FL Job structure created at: $FL_JOB_DIR"
tree -L 3 "$FL_JOB_DIR" 2>/dev/null || ls -R "$FL_JOB_DIR"
echo ""

# ============================================================================
# Create Workspace Configuration
# ============================================================================
echo "========================================="
echo "Creating Workspace Configuration..."
echo "========================================="

WORKSPACE_CONFIG="$PROJECT_ROOT/workspace_config.json"

cat > "$WORKSPACE_CONFIG" << EOF
{
  "format_version": 2,
  "clients": [
    {
      "name": "site-1"
    },
    {
      "name": "site-2"
    }
  ]
}
EOF

echo "✓ Workspace configuration created: $WORKSPACE_CONFIG"
echo ""

# ============================================================================
# Create Client Privacy Configs
# ============================================================================
echo "========================================="
echo "Creating Client Privacy Configs..."
echo "========================================="

# These will be used by simulator
mkdir -p "$PROJECT_ROOT/client_configs/site-1"
mkdir -p "$PROJECT_ROOT/client_configs/site-2"

for SITE in "site-1" "site-2"; do
    cat > "$PROJECT_ROOT/client_configs/$SITE/privacy.json" << 'EOF'
{
  "client": {
    "allow_adhoc_dataset": true
  }
}
EOF
    echo "✓ Privacy config created for $SITE"
done

echo ""

# ============================================================================
# Clean Previous Results
# ============================================================================
echo "========================================="
echo "Cleaning Previous Results..."
echo "========================================="

rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"
echo "✓ Workspace cleaned: $WORKSPACE_DIR"
echo ""

# ============================================================================
# Run Simulator
# ============================================================================
echo "========================================="
echo "Starting NVFLARE Simulator..."
echo "========================================="
echo ""
echo "Simulator Configuration:"
echo "  Job:       $FL_JOB_DIR"
echo "  Workspace: $WORKSPACE_DIR"
echo "  Clients:   2 (site-1, site-2)"
echo "  Threads:   2"
echo ""
echo "Starting training..."
echo ""

cd "$PROJECT_ROOT"

nvflare simulator \
    "$FL_JOB_DIR" \
    -w "$WORKSPACE_DIR" \
    -n 2 \
    -t 2

echo ""
echo "========================================="
echo "Simulator Finished!"
echo "========================================="
echo ""
echo "Results saved to: $WORKSPACE_DIR"
echo ""
echo "View logs:"
echo "  Server:  cat $WORKSPACE_DIR/server/log.txt"
echo "  Site-1:  cat $WORKSPACE_DIR/site-1/log.txt"
echo "  Site-2:  cat $WORKSPACE_DIR/site-2/log.txt"
echo ""
echo "Global model: $WORKSPACE_DIR/server/global_model.pt"
echo ""
