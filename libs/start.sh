#!/bin/zsh

TARGET_DIR = '/Volumes/My Passport/Sasaki/MTsingleBeads'

# tifに変換する
pixi run python nd2_to_tif_8bit.py \
    '${TARGET_DIR}/beads_trans_crop_crop.nd2' \
    '${TARGET_DIR}/20260121/beads_trans_crop_crop/TRITC_hist_match_prev' \
    --channel TRITC \
    --mode hist_match_prev

# zarrに変換する (MTs)
pixi run python nd2_to_zarr_channel.py \
    --file_path '${TARGET_DIR}/beads_trans_crop_crop.nd2' \
    --out_dir '${TARGET_DIR}/20260121/beads_trans_crop_crop' \
    --channel TRITC \
    --out_name TRITC.zarr

# zarrに変換する (beads)
pixi run python nd2_to_zarr_channel.py \
    --file_path '${TARGET_DIR}/beads_trans_crop_crop.nd2' \
    --out_dir '${TARGET_DIR}/20260121/beads_trans_crop_crop' \
    --sigma '(0, 2, 2)' \
    --channel Cy5 \
    --out_name Cy5.zarr

# nematic parameterの計算
pixi run calcAFT.py \
    '${TARGET_DIR}/TRITC.zarr' \
    --n_jobs 4