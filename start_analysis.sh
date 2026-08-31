#!/bin/zsh

# Base data directory (Auto-detect macOS or Linux NAS mount)
if [ -d "/Volumes/data/Sasaki/MTsingleBeads" ]; then
    ROOT_DIR="/Volumes/data/Sasaki/MTsingleBeads"
elif [ -d "/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads" ]; then
    ROOT_DIR="/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads"
else
    ROOT_DIR="/Volumes/data/Sasaki/MTsingleBeads"
fi

# Bead conditions and their corresponding particle masking radii (in pixels)
# Format: "bead_folder:particle_radius"
CONDITIONS=(
    "beads06um:3"
    "beads1um:6"
    "beads3um:15"
    "beads5um:25"
    "beads7um:35"
    "beads20um:95"
)

echo "=========================================="
echo "Starting batch analysis on: $ROOT_DIR"
echo "=========================================="

for COND in "${CONDITIONS[@]}"; do
    BEAD_NAME="${COND%%:*}"
    RADIUS="${COND##*:}"
    TARGET_DIR="${ROOT_DIR}/${BEAD_NAME}"

    if [ ! -d "$TARGET_DIR" ]; then
        echo "[SKIP] Directory not found: $TARGET_DIR"
        continue
    fi

    echo "\n>>> Processing condition: $BEAD_NAME (particle_radius=$RADIUS)"

    # Look for .nd2 files or subdirectories
    for FILE in "${TARGET_DIR}"/*/*.nd2 "${TARGET_DIR}"/*/*/*.nd2; do
        [ -f "$FILE" ] || continue
        DIRNAME=$(dirname "$FILE")
        BASENAME=$(basename "$FILE")
        EXP_DIR="${DIRNAME}/${BASENAME%.nd2}"

        # If directory doesn't exist, check DIRNAME directly
        if [ ! -d "$EXP_DIR" ]; then
            EXP_DIR="$DIRNAME"
        fi

        if [ -f "${EXP_DIR}/GFP_flows.h5" ] && [ -f "${EXP_DIR}/beads_tracks.csv" ]; then
            echo "Processing: $EXP_DIR"

            # 1. Local polar order analysis
            pixi run python libs/calc_bg_polar.py "$EXP_DIR"
            pixi run python libs/calc_local_polar.py "$EXP_DIR" --particle_radius "$RADIUS"

            # 2. Angular spatial correlation analysis
            pixi run python libs/calc_bg_angular_correlation.py "$EXP_DIR"
            pixi run python libs/calc_angular_spatial_correlation.py "$EXP_DIR" --particle_radius "$RADIUS"
        fi
    done
done

echo "\n=========================================="
echo "All batch analyses completed!"
echo "=========================================="