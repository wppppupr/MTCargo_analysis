TARGET_DIR='/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp'
TARGET_DIR1='/Volumes/My Passport/Sasaki/MTsingleBeads/20260121/beads_trans_crop_crop'
TARGET_DIR2='/Volumes/My Passport/Sasaki/MTsingleBeads/20260121/exp_crop1'

pixi run python libs/calc_local_polar.py "${TARGET_DIR}"
pixi run python libs/calc_bg_polar.py "${TARGET_DIR}"

pixi run python libs/calc_local_polar.py "${TARGET_DIR1}"
pixi run python libs/calc_bg_polar.py "${TARGET_DIR1}"

pixi run python libs/calc_local_polar.py "${TARGET_DIR2}"
pixi run python libs/calc_bg_polar.py "${TARGET_DIR2}"