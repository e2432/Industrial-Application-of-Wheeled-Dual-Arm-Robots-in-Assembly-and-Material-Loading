# RobotA 一键建图脚本

## 建图运动方式

不要只让机器人原地转一圈。建议按以下顺序操作：

1. 确认周围没有人员和障碍，急停按钮可以随时使用。
2. 先原地低速旋转约一圈，让 SLAM 获得四周初始结构。
3. 沿场地边缘或主要通道缓慢行驶一圈，中途转弯要慢。
4. 尽量回到出发点，并保持接近出发时的朝向，以便产生闭环。
5. 在 RViz 中检查墙面没有严重重影、断裂或跳变后再保存地图。

## 安装

把 `robot_a_mapping.sh` 复制到机器人：

```bash
cp robot_a_mapping.sh /home/niic/robot_a_mapping.sh
chmod +x /home/niic/robot_a_mapping.sh
```

如果没有 `topic_tools`，只需安装一次：

```bash
sudo apt install -y ros-humble-topic-tools
```

## 一键启动

```bash
sudo bash /home/niic/robot_a_mapping.sh start
```

脚本会完成：

- 加载 ROS 2 Humble、RobotA 与建图工作空间环境；
- 停止会与建图冲突的旧定位和旧 Nav2；
- 检查麦克纳姆底盘控制器与轮速里程计；
- 启动或复用 Livox PointCloud2 驱动；
- 将底盘 `tf_odometry` 转发到标准 `/tf`；
- 发布 `base_link -> base_footprint` 静态变换；
- 把 `/livox/lidar` 转换为 `/scan`；
- 使用 `start_hardware:=false` 启动 SLAM Toolbox 和建图 RViz；
- 验证 `/scan`、`/map` 和关键 TF。

如果脚本需要自行启动 Livox，Livox 官方 PointCloud2 入口可能会同时打开一个点云 RViz。建图 RViz 出现后，可关闭点云显示窗口，保留固定坐标系为 `map` 的建图窗口。

## 查看状态

```bash
sudo bash /home/niic/robot_a_mapping.sh status
```

## 低速遥控

另开 root 终端：

```bash
source /opt/robot/current/bin/ros2env
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args \
  -r cmd_vel:=/mecanum_drive_controller/reference_unstamped
```

启动后先多按几次 `z` 和 `x` 降低线速度、角速度。空格键用于立即发送停止指令。

## 保存地图

地图名只能包含英文字母、数字、下划线和短横线：

```bash
sudo bash /home/niic/robot_a_mapping.sh save factory_floor_01
```

地图保存在：

```text
/home/niic/maps/factory_floor_01.yaml
/home/niic/maps/factory_floor_01.pgm
```

## 停止建图

```bash
sudo bash /home/niic/robot_a_mapping.sh stop
```

停止时脚本会先向底盘发送一次零速度，再停止 SLAM、点云转换、辅助 TF 和由脚本启动的 Livox。脚本不会停止系统原本已经运行的 Livox，也不会自动恢复旧定位/Nav2。

## 日志

每次启动的日志保存在：

```text
/root/.ros/log/robot_a_mapping_日期_时间/
```

如果启动失败，优先查看脚本打印出的具体日志路径。
