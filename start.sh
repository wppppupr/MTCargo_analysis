#!/bin/bash
TARGET_DIR='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads5um/20260715'

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

cd ..
cd opticalflow-acivenematics

for FILE in "${TARGET_DIR}"/*; do
    [ -d "$FILE" ] || continue

    BASENAME=$(basename "$FILE")
    echo "Processing $BASENAME..."
    pixi run python raft_finetune/inference.py \
        --image_dir $FILE/GFP \
        --model_path /home/sasaki/opticalflow-activenematics/raft_finetune/model/raft_finetune_epoch_5.pth \
        --output_h5 $FILE/GFP_flows.h5 \
        --tile_size 1024

    pixi run python raft_finetune/calc_velocity.py \
        --h5_path $FILE/GFP_flows.h5 \
        --out_h5 $FILE/velocities.h5 \
        --mean_only
done

echo "Done"