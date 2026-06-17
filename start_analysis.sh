TARGET_DIR='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads5um/20260616'
H5_FILE='GFP_flows.h5'

for FILE in "${TARGET_DIR}"/*; do
    [ -d "$FILE" ] || continue

    BASENAME=$(basename "$FILE")

    echo "Processing $BASENAME..."

    pixi run python libs/calc_local_polar.py "${TARGET_DIR}/${BASENAME%.nd2}" --particle_radius 30 --h5_file "${H5_FILE}"
    pixi run python libs/calc_bg_polar.py "${TARGET_DIR}/${BASENAME%.nd2}" --h5_file "${H5_FILE}"

done