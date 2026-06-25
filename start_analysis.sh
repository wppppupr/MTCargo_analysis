TARGET_DIR='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/control/MTs6uM/20260625'
TARGET_DIR2='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/control/MTs10uM/20260625'
TARGET_DIR5='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads7um/20260608'
TARGET_DIR6='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/control/MTs8uM/20260624'
H5_FILE='GFP_flows.h5'


for FILE in "${TARGET_DIR}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    echo "Processing $BASENAME..."

    # zarrに変換する (MTs)
    pixi run python libs/nd2_to_zarr_channel.py \
        --file_path "$FILE" \
        --out_dir "${TARGET_DIR}/${BASENAME%.nd2}" \
        --channel GFP \
        --out_name "GFP.zarr"

    # nematic parameterの計算
    pixi run python libs/calcAFT.py \
        "${TARGET_DIR}/${BASENAME%.nd2}" \
        --zarr_path "GFP.zarr" \
        --neighborhood_radius 5

done

for FILE in "${TARGET_DIR2}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    echo "Processing $BASENAME..."

    # zarrに変換する (MTs)
    pixi run python libs/nd2_to_zarr_channel.py \
        --file_path "$FILE" \
        --out_dir "${TARGET_DIR2}/${BASENAME%.nd2}" \
        --channel GFP \
        --out_name "GFP.zarr"

    # nematic parameterの計算
    pixi run python libs/calcAFT.py \
        "${TARGET_DIR2}/${BASENAME%.nd2}" \
        --zarr_path "GFP.zarr" \
        --neighborhood_radius 5

done

echo "Done"

for FILE in "${TARGET_DIR5}"/*; do
    [ -d "$FILE" ] || continue

    BASENAME=$(basename "$FILE")

    echo "Processing $BASENAME..."

    pixi run python libs/calc_local_polar.py "${TARGET_DIR5}/${BASENAME%.nd2}" --particle_radius 33 --h5_file "${H5_FILE}"
    pixi run python libs/calc_bg_polar.py "${TARGET_DIR5}/${BASENAME%.nd2}" --h5_file "${H5_FILE}"

done

for FILE in "${TARGET_DIR6}"/*; do
    [ -d "$FILE" ] || continue

    BASENAME=$(basename "$FILE")

    echo "Processing $BASENAME..."

    pixi run python libs/calc_local_polar_noCargo.py "${TARGET_DIR6}/${BASENAME%.nd2}" --h5_file "${H5_FILE}"

done