TARGET_DIR='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads1uM/20260121'
TARGET_DIR2='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads1uM/20260122'
TARGET_DIR3='/Volumes/My Passport/Sasaki/MTSingleBeads/beads7um/20260608'
TARGET_DIR4='/Volumes/My Passport/Sasaki/MTSingleBeads/beads3um'
TARGET_DIR5='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads5uM/20260616'
TARGET_DIR6='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/control/MTs8uM/20260624'

H5_FILE='GFP_flows.h5'

<<co
pixi run python libs/bead_flow_interaction.py \
    "${TARGET_DIR}/beads_trans_crop_crop" \
    --h5_file "${H5_FILE}" \
    --particle_radius 6

pixi run python libs/bead_flow_interaction_gaussian.py \
    "${TARGET_DIR}/beads_trans_crop_crop" \
    --h5_file "${H5_FILE}" \
    --particle_radius 6

pixi run python libs/calc_local_polar.py \
    "${TARGET_DIR}/beads_trans_crop_crop" \
    --h5_file "${H5_FILE}" \
    --particle_radius 6

pixi run python libs/calc_local_polar_gaussian.py \
    "${TARGET_DIR}/beads_trans_crop_crop" \
    --h5_file "${H5_FILE}" \
    --particle_radius 6

pixi run python libs/calc_bg_polar.py \
    "${TARGET_DIR}/beads_trans_crop_crop" \
    --h5_file "${H5_FILE}"

pixi run python libs/calc_bg_polar_gaussian.py \
    "${TARGET_DIR}/beads_trans_crop_crop" \
    --h5_file "${H5_FILE}"

pixi run python libs/bead_flow_interaction.py \
    "${TARGET_DIR}/exp_crop1" \
    --h5_file "${H5_FILE}" \
    --particle_radius 6

pixi run python libs/bead_flow_interaction_gaussian.py \
    "${TARGET_DIR}/exp_crop1" \
    --h5_file "${H5_FILE}" \
    --particle_radius 6

pixi run python libs/calc_local_polar.py \
    "${TARGET_DIR}/exp_crop1" \
    --h5_file "${H5_FILE}" \
    --particle_radius 6

pixi run python libs/calc_local_polar_gaussian.py \
    "${TARGET_DIR}/exp_crop1" \
    --h5_file "${H5_FILE}" \
    --particle_radius 6

pixi run python libs/calc_bg_polar.py \
    "${TARGET_DIR}/exp_crop1" \
    --h5_file "${H5_FILE}"

pixi run python libs/calc_bg_polar_gaussian.py \
    "${TARGET_DIR}/exp_crop1" \
    --h5_file "${H5_FILE}"

for FILE in "${TARGET_DIR2}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    pixi run python libs/bead_flow_interaction.py \
        "${TARGET_DIR2}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 6

    pixi run python libs/bead_flow_interaction_gaussian.py \
        "${TARGET_DIR2}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 6

    pixi run python libs/calc_local_polar.py \
        "${TARGET_DIR2}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 6

    pixi run python libs/calc_local_polar_gaussian.py \
        "${TARGET_DIR2}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 6

    pixi run python libs/calc_bg_polar.py \
        "${TARGET_DIR2}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}"

    pixi run python libs/calc_bg_polar_gaussian.py \
        "${TARGET_DIR2}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}"

done
co

for FILE in "${TARGET_DIR3}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    pixi run python libs/bead_flow_interaction.py \
        "${TARGET_DIR3}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 33

    pixi run python libs/bead_flow_interaction_gaussian.py \
        "${TARGET_DIR3}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 33

    pixi run python libs/calc_local_polar.py \
        "${TARGET_DIR3}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 33

    pixi run python libs/calc_local_polar_gaussian.py \
        "${TARGET_DIR3}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 33

    pixi run python libs/calc_bg_polar.py \
        "${TARGET_DIR3}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}"

    pixi run python libs/calc_bg_polar_gaussian.py \
        "${TARGET_DIR3}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}"

done



for FILE in "${TARGET_DIR4}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    pixi run python libs/bead_flow_interaction.py \
        "${TARGET_DIR4}/${BASENAME%.nd2}" \
        --h5_file "TRITC_flows.h5" \
        --particle_radius 16

    pixi run python libs/bead_flow_interaction_gaussian.py \
        "${TARGET_DIR4}/${BASENAME%.nd2}" \
        --h5_file "TRITC_flows.h5" \
        --particle_radius 16

    pixi run python libs/calc_local_polar.py \
        "${TARGET_DIR4}/${BASENAME%.nd2}" \
        --h5_file "TRITC_flows.h5" \
        --particle_radius 16

    pixi run python libs/calc_local_polar_gaussian.py \
        "${TARGET_DIR4}/${BASENAME%.nd2}" \
        --h5_file "TRITC_flows.h5" \
        --particle_radius 16

    pixi run python libs/calc_bg_polar.py \
        "${TARGET_DIR4}/${BASENAME%.nd2}" \
        --h5_file "TRITC_flows.h5"

    pixi run python libs/calc_bg_polar_gaussian.py \
        "${TARGET_DIR4}/${BASENAME%.nd2}" \
        --h5_file "TRITC_flows.h5"

done

<<co

for FILE in "${TARGET_DIR5}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    pixi run python libs/bead_flow_interaction.py \
        "${TARGET_DIR5}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 23

    pixi run python libs/bead_flow_interaction_gaussian.py \
        "${TARGET_DIR5}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 23
    
    pixi run python libs/calc_local_polar.py \
        "${TARGET_DIR5}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 23

    pixi run python libs/calc_local_polar_gaussian.py \
        "${TARGET_DIR5}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}" \
        --particle_radius 23

    pixi run python libs/calc_bg_polar.py \
        "${TARGET_DIR5}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}"

    pixi run python libs/calc_bg_polar_gaussian.py \
        "${TARGET_DIR5}/${BASENAME%.nd2}" \
        --h5_file "${H5_FILE}"

done

for FILE in "${TARGET_DIR6}"/*.nd2; do
    BASENAME=$(basename "$FILE")
    pixi run python libs/calc_local_polar_noCargo.py "${TARGET_DIR6}/${BASENAME%.nd2}"
done
co