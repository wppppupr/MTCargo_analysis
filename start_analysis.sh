TARGET_DIR='/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp001'

pixi run python libs/visualize_flow_video.py \
    --image_dir "${TARGET_DIR}/GFP"\
    --h5_path "${TARGET_DIR}/GFP_flows.h5"\
    --output_video "${TARGET_DIR}/flow_movie.mp4"


TARGET_DIR='/Volumes/My Passport/Sasaki/MTsingleBeads/20260121/beads_trans_crop_crop'

pixi run python libs/visualize_flow_video.py \
    --image_dir "${TARGET_DIR}/GFP"\
    --h5_path "${TARGET_DIR}/GFP_flows.h5"\
    --output_video "${TARGET_DIR}/flow_movie.mp4"


TARGET_DIR='/Volumes/My Passport/Sasaki/MTsingleBeads/20260121/exp_crop1'

pixi run python libs/visualize_flow_video.py \
    --image_dir "${TARGET_DIR}/GFP"\
    --h5_path "${TARGET_DIR}/GFP_flows.h5"\
    --output_video "${TARGET_DIR}/flow_movie.mp4"