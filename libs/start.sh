#!/bin/zsh

TARGET_DIR="/Volumes/My Passport/Sasaki/MTsingleBeads/beads3um/20260416"
ND2="exp002.nd2"

# 変数展開を有効にするため、ダブルクォーテーションで囲む
# tifに変換する
pixi run python nd2_to_tif_8bit.py \
    "${TARGET_DIR}/${ND2}" \
    "${TARGET_DIR}/TRITC_hist_match_prev" \
    --channel TRITC \
    --mode hist_match_prev

# zarrに変換する (MTs)
pixi run python nd2_to_zarr_channel.py \
    --file_path "${TARGET_DIR}/${ND2}" \
    --out_dir "${TARGET_DIR}" \
    --channel TRITC \
    --out_name TRITC.zarr

# zarrに変換する (beads)
pixi run python nd2_to_zarr_channel.py \
    --file_path "${TARGET_DIR}/${ND2}" \
    --out_dir "${TARGET_DIR}" \
    --sigma '(0,2,2)'\
    --channel 1 \
    --out_name beads.zarr

# nematic parameterの計算
pixi run python calcAFT.py \
    "${TARGET_DIR}" \
    --zarr_path "TRITC.zarr" \
    --n_jobs 4