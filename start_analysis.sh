#TARGET_DIR='/Volumes/data/Sasaki/MTsingleBeads/control/MTs8uM/20260624'
TARGET_DIR='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads06um/20260716'

for FILE in "${TARGET_DIR}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    pixi run python libs/calc_bg_polar.py \
    "${TARGET_DIR}/${BASENAME%.nd2}" 

    pixi run python libs/calc_local_polar.py \
    "${TARGET_DIR}/${BASENAME%.nd2}" \
    --particle_radius 25

done