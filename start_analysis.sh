TARGET_DIR='/mnt/SSD/Sasaki/MTsingleBeads/20260122/exp'
TARGET_DIR0='/mnt/SSD/Sasaki/MTsingleBeads/20260122/exp001'
TARGET_DIR1='/mnt/SSD/Sasaki/MTsingleBeads/20260121/beads_trans_crop_crop'
TARGET_DIR2='/mnt/SSD/Sasaki/MTsingleBeads/20260121/exp_crop1'

#pixi run python libs/optical_flow.py "${TARGET_DIR}/GFP" "${TARGET_DIR}/Farneback.h5"
pixi run python libs/optical_flow.py "${TARGET_DIR0}/GFP" "${TARGET_DIR0}/Farneback.h5"
pixi run python libs/optical_flow.py "${TARGET_DIR1}/GFP" "${TARGET_DIR1}/Farneback.h5"
pixi run python libs/optical_flow.py "${TARGET_DIR2}/GFP" "${TARGET_DIR2}/Farneback.h5"

<<comment
pixi run python libs/calc_local_polar.py "${TARGET_DIR}"
pixi run python libs/calc_bg_polar.py "${TARGET_DIR}"

pixi run python libs/calc_local_polar.py "${TARGET_DIR1}"
pixi run python libs/calc_bg_polar.py "${TARGET_DIR1}"

pixi run python libs/calc_local_polar.py "${TARGET_DIR2}"
pixi run python libs/calc_bg_polar.py "${TARGET_DIR2}"
comment