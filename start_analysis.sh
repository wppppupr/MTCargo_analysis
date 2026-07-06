TARGET_DIR='/Volumes/data/Sasaki/MTsingleBeads/control/MTs8uM/20260624'

H5_FILE='GFP_flows.h5'

for FILE in "${TARGET_DIR}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    pixi run python libs/calc_strain_rate.py \
    "${TARGET_DIR}/${BASENAME%.nd2}/${H5_FILE}" \
    "${TARGET_DIR}/${BASENAME}/vortex.h5"

done