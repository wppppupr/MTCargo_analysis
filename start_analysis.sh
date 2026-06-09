TARGET_DIR='/Volumes/data/Sasaki/MTsingleBeads/beads7um/20260608'
H5_FILE='TRITC_flows.h5'

for FILE in "${TARGET_DIR}"/*; do
    [ -d "$FILE" ] || continue

    BASENAME=$(basename "$FILE")

    echo "Processing $BASENAME..."

    pixi run python libs/calc_local_polar.py "${TARGET_DIR}/${BASENAME%.nd2}" --particle_radius 36 --h5_file "${H5_FILE}"
    pixi run python libs/calc_bg_polar.py "${TARGET_DIR}/${BASENAME%.nd2}" --h5_file "${H5_FILE}"

done