TARGET_DIR='/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp001'
TARGET_DIR1='/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp'
TARGET_DIR2='/Volumes/My Passport/Sasaki/MTsingleBeads/20260121/exp_crop1'


pixi run python libs/optical_flow.py \
    "${TARGET_DIR}/GFP"\
    "${TARGET_DIR}/Farneback.h5"

pixi run python libs/evaluate_of.py \
    --h5_path "${TARGET_DIR}/Farneback.h5" \
    --csv_path "${TARGET_DIR}/MTtrack.csv"


pixi run python libs/optical_flow.py \
    "${TARGET_DIR1}/GFP"\
    "${TARGET_DIR1}/Farneback.h5"

pixi run python libs/evaluate_of.py \
    --h5_path "${TARGET_DIR1}/Farneback.h5" \
    --csv_path "${TARGET_DIR1}/MTtrack.csv"

pixi run python libs/optical_flow.py \
    "${TARGET_DIR2}/GFP"\
    "${TARGET_DIR2}/Farneback.h5"

pixi run python libs/evaluate_of.py \
    --h5_path "${TARGET_DIR2}/Farneback.h5" \
    --csv_path "${TARGET_DIR2}/MTtrack.csv"