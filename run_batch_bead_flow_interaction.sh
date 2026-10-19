#!/usr/bin/env bash
set -e

# activate pixi env or run via pixi
PIXI_PYTHON="/home/sasaki/MTCargo_analysis/.pixi/envs/default/bin/python"

declare -A RAD_MAP=(
    ["beads06um"]="3"
    ["beads1um"]="6"
    ["beads3um"]="15"
    ["beads5um"]="25"
    ["beads7um"]="35"
    ["beads20um"]="95"
)

TARGET_DIRS=(
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um001"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um002"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um003"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um001_crop"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um002"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um003"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um_crop"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um001"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um002"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um003"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um001"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um002"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um003"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um002"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um003"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um001"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um002"
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um_crop"
)

TOTAL=${#TARGET_DIRS[@]}
IDX=0

echo "=== Starting beads_flow_interaction batch for $TOTAL experiments ==="

for EXP_DIR in "${TARGET_DIRS[@]}"; do
    IDX=$((IDX + 1))
    # Determine condition from path
    COND=""
    for C in beads06um beads1um beads3um beads5um beads7um beads20um; do
        if [[ "$EXP_DIR" == *"/$C/"* ]]; then
            COND="$C"
            break
        fi
    done

    PRAD="${RAD_MAP[$COND]}"
    echo "[$IDX/$TOTAL] Processing ($COND, particle_radius=$PRAD): $EXP_DIR"

    $PIXI_PYTHON libs/bead_flow_interaction.py \
        "$EXP_DIR" \
        --particle_radius "$PRAD"

    echo "[$IDX/$TOTAL] Completed: $EXP_DIR"
done

echo "=== All $TOTAL experiments finished successfully! ==="
