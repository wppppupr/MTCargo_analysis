TARGET_DIR='/mnt/SSD/Sasaki/MTsingleBeads/20260122/exp'
H5_FILE='TRITC_flows.h5'

for FILE in "${TARGET_DIR}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    echo "Processing $BASENAME..."

    pixi run python libs/calc_local_polar.py "${TARGET_DIR}/${BASENAME%.nd2}" --particle_radius 6 --h5_file "${H5_FILE}"
    pixi run python libs/calc_bg_polar.py "${TARGET_DIR}/${BASENAME%.nd2}" --h5_file "${H5_FILE}"

done