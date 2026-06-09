#!/bin/bash

TARGET_DIR='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads7um/20260608'
TARGET_DIR2='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads8um/20260608'

# zarrに変換する (beads)
pixi run python libs/nd2_to_zarr_channel.py \
    --file_path "/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads8um/20260608/beads8um.nd2" \
    --out_dir /mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads8um/20260608/beads8um \
    --sigma '(0,2,2)' \
    --channel Cy5 \
    --out_name "beads.zarr"

<<co

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

co