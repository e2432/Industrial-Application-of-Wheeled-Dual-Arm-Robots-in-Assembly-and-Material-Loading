# competition_mission 部署说明

## 固定任务路线

```text
start -> point1 -> point2
-> grab1（抓取/抬升）-> drop1（放下/复位）-> grab1（退回）
-> grab2（抓取/抬升）-> drop2（放下/复位）-> grab2（退回）
-> grab3（抓取/抬升）-> drop3（放下/复位）-> grab3（退回）
-> grab4（抓取/抬升）-> drop4（放下/复位）-> grab4（退回）
-> point2 -> point1 -> start
```

退回 `grabN` 时只导航，不重复抓取。路点坐标来自工作区原 `real_waypoints.yaml`；新配置使用唯一 ID，并修复了原文件 `drop3` 的缩进。

## 复制到原机器

把文件按下面的原名和路径复制：

```text
competition_mission.sh
  -> /home/niic/Desktop/competition_mission.sh

competition_mission.py
box_waypoint_mission.py
bridge_xyz_offset_box_wide_descend_postlift.py
  -> /home/niic/Desktop/competition/vision/box/

competition_waypoints.yaml
  -> /home/niic/robot_a_ws/src/robot_a_real_navigation/config/
```

然后执行：

```bash
chmod +x /home/niic/Desktop/competition_mission.sh
```

其中 `box_waypoint_mission.py` 和抓取脚本是两套任务共同使用的公共版本。部署后，原 `m1.sh` 和新 `competition_mission.sh` 都从同一路径调用它们。

```text
m1.sh -> box_waypoint_mission.py -> bridge_xyz_offset_box_wide_descend_postlift.py
competition_mission.sh -> competition_mission.py -> box_waypoint_mission.py
                                            -> bridge_xyz_offset_box_wide_descend_postlift.py
```

不要用 `competition_mission.sh` 替换或重命名 `m1.sh`。原 `m1.sh` 仍使用 `real_waypoints.yaml` 执行原路线；新入口使用 `competition_waypoints.yaml` 执行比赛路线。两者共享底层实现，但任务入口和路点配置彼此独立。

## 运行前设置地图

编辑 `/home/niic/Desktop/competition_mission.sh` 中的：

```bash
MAP_FILE="__MAP_FILE__"
```

将占位符换成原机地图 YAML 的绝对路径。未替换时，脚本会在启动机器人运动之前退出。

## 原机继续依赖的现有文件

新任务仍使用原机 `/home/niic/Desktop/competition/vision/box` 目录中的视觉、抬升、释放、复位脚本及抓取脚本的同目录依赖：

```text
box_ros2_robust.py
bridge_xyz_offset_box_common_height.py（以及其原有依赖）
box_dual_arm_place_height_constrained_v4_smooth.py
box_dual_arm_release_outward_3cm.py
box_all_joints_zero.py
```

## 运行新任务

```bash
sudo -i
cd /home/niic/Desktop
./competition_mission.sh
```

两个任务共同使用当前视觉纠错、IK 底盘纠错、速度仲裁、Nav2 取消处理、TF 稳定等待及代价地图刷新逻辑。比赛任务只增加路线编排，不复制这些底层实现。
