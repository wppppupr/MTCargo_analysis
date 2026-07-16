#TARGET_DIR='/Volumes/data/Sasaki/MTsingleBeads/control/MTs8uM/20260624'
TARGET_DIR='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads5um/20260715'

cd ..
cd opticalflow-activenematics

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

cd ..
cd MTCargo_analysis

for FILE in "${TARGET_DIR}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    pixi run python libs/calc_bg_polar.py \
    "${TARGET_DIR}/${BASENAME%.nd2}" 

    pixi run python libs/calc_local_polar.py \
    "${TARGET_DIR}/${BASENAME%.nd2}" \
    --particle_radius 25

done