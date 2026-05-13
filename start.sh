#!/bin/zsh

TARGET_DIR='/Volumes/My Passport/Sasaki/MTsingleBeads/20260512'
ND2="beads3um001.nd2"

<<comment
# 変数展開を有効にするため、ダブルクォーテーションで囲む
# tifに変換する
pixi run python libs/nd2_to_tif_8bit.py \
    "${TARGET_DIR}/${ND2}" \
    "${TARGET_DIR}/GFP" \
    --channel GFP \

# zarrに変換する (MTs)
pixi run python libs/nd2_to_zarr_channel.py \
    --file_path "${TARGET_DIR}/${ND2}" \
    --out_dir "${TARGET_DIR}" \
    --channel GFP \
    --out_name GFP.zarr

# zarrに変換する (beads)
pixi run python libs/nd2_to_zarr_channel.py \
    --file_path "${TARGET_DIR}/${ND2}" \
    --out_dir "${TARGET_DIR}" \
    --sigma '(0,2,2)'\
    --channel Cy5 \
    --out_name beads.zarr
comment

# nematic parameterの計算
pixi run python libs/calcAFT.py \
    "${TARGET_DIR}" \
    --zarr_path "GFP.zarr" \
    --n_jobs 4

