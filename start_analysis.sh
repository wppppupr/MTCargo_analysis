#TARGET_DIR='/Volumes/data/Sasaki/MTsingleBeads/control/MTs8uM/20260624'
TARGET_DIR='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads1um/20260707'
TARGET_DIR2='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/beads3um/20260708'

for FILE in "${TARGET_DIR}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    pixi run python libs/calc_bg_polar.py \
    "${TARGET_DIR}/${BASENAME%.nd2}" 

    pixi run python libs/calc_local_polar.py \
    "${TARGET_DIR}/${BASENAME%.nd2}" \
    --particle_radius 6

done

for FILE in "${TARGET_DIR2}"/*.nd2; do
    BASENAME=$(basename "$FILE")

    pixi run python libs/calc_bg_polar.py \
    "${TARGET_DIR2}/${BASENAME%.nd2}" 

    pixi run python libs/calc_local_polar.py \
    "${TARGET_DIR2}/${BASENAME%.nd2}" \
    --particle_radius 16

done