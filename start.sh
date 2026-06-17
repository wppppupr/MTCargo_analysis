#!/bin/bash

TARGET_DIR='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads5um/20260616'

for FILE in "${TARGET_DIR}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    echo "Processing $BASENAME..."

    # tifに変換
    pixi run python libs/nd2_to_tif_8bit.py \
        "$FILE" \
        "${TARGET_DIR}/${BASENAME%.nd2}/GFP" \
        --channel GFP

    # zarrに変換する (MTs)
    pixi run python libs/nd2_to_zarr_channel.py \
        --file_path "$FILE" \
        --out_dir "${TARGET_DIR}/${BASENAME%.nd2}" \
        --channel GFP \
        --out_name "GFP.zarr"

    # zarrに変換する (beads)
    pixi run python libs/nd2_to_zarr_channel.py \
        --file_path "$FILE" \
        --out_dir "${TARGET_DIR}/${BASENAME%.nd2}" \
        --sigma '(0,2,2)' \
        --channel Cy5 \
        --out_name "beads.zarr"

    # nematic parameterの計算
    pixi run python libs/calcAFT.py \
        "${TARGET_DIR}/${BASENAME%.nd2}" \
        --zarr_path "GFP.zarr" \
        --neighborhood_radius 5

done

echo "Done"