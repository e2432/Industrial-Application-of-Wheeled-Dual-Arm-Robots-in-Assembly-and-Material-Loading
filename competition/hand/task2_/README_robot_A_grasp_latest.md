safety check:
/usr/bin/python3.10 ik_grasp_dual_mirror_real_v4.py \
  -x -0.15 \
  -y 0.79 \
  -z 0.94 \
  --velocity-scale 0.30

run:
/usr/bin/python3.10 ik_grasp_dual_mirror_real_v4.py \
  -x -0.24 \
  -y 0.79 \
  -z 0.78 \
  --velocity-scale 0.30 \
  --execute

gripper:
/usr/bin/python3.10 ik_grasp_dual_mirror_real_v4.py \
  -x -0.24 \
  -y 0.79 \
  -z 0.68 \
  --velocity-scale 0.30 \
  --execute \
  --close-gripper

