#!/bin/zsh

pixi run python nd2_to_tif_8bit.py \
    '/Volumes/My Passport/Sasaki/MTsingleBeads/20260121/beads_trans_crop_crop/beads_trans_crop_crop.nd2' \
    '/Volumes/My Passport/Sasaki/MTsingleBeads/20260121/beads_trans_crop_crop/GFP_hist_match_prev' \
    --channel GFP \
    --mode hist_match_prev

pixi run python nd2_to_tif_8bit.py \
    '/Volumes/My Passport/Sasaki/MTsingleBeads/20260121/exp_crop1/exp_crop1.nd2' \
    '/Volumes/My Passport/Sasaki/MTsingleBeads/20260121/exp_crop1/GFP_hist_match_prev' \
    --channel GFP \
    --mode hist_match_prev

pixi run python nd2_to_tif_8bit.py \
    '/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp/exp_crop.nd2' \
    '/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp/GFP_hist_match_prev' \
    --channel GFP \
    --mode hist_match_prev

pixi run python nd2_to_tif_8bit.py \
    '/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp001/exp001.nd2' \
    '/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp001/GFP_hist_match_prev' \
    --channel GFP \
    --mode hist_match_prev