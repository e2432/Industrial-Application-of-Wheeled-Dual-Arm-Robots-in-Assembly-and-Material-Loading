#!/usr/bin/env bash
# RobotA real-robot mapping session manager (ROS 2 Humble).
# Commands:
#   sudo bash robot_a_mapping.sh start
#   sudo bash robot_a_mapping.sh status
#   sudo bash robot_a_mapping.sh save map_name
#   sudo bash robot_a_mapping.sh stop

set -Eeuo pipefail

ACTION="${1:-start}"
MAP_NAME="${2:-}"

ROBOT_SETUP="/opt/robot/current/bin/ros2env"
ROS_SETUP="/opt/ros/humble/setup.bash"
NAV_SETUP="/home/niic/robot_a_ws/install/setup.bash"
STATE_DIR="/run/robot_a_mapping"
SESSION_FILE="${STATE_DIR}/session_log_dir"
MAP_DIR="/home/niic/maps"
START_IN_PROGRESS=0
START_COMPLETED=0

LIVOX_NODE="/livox_lidar_publisher"
SCAN_NODE="/livox_to_scan"
SLAM_NODE="/slam_toolbox"
TF_RELAY_NODE="/wheel_tf_relay"
FOOTPRINT_TF_NODE="/base_footprint_static_tf"

die() {
  echo "[错误] $*" >&2
  exit 1
}

note() {
  echo "[信息] $*"
}

ok() {
  echo "[正常] $*"
}

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    die "请使用：sudo bash $0 ${ACTION}${MAP_NAME:+ ${MAP_NAME}}"
  fi
}

load_ros_environment() {
  [[ -r "$ROS_SETUP" ]] || die "找不到 $ROS_SETUP"
  [[ -r "$ROBOT_SETUP" ]] || die "找不到 $ROBOT_SETUP"
  [[ -r "$NAV_SETUP" ]] || die "找不到 $NAV_SETUP，请先编译 robot_a_real_navigation"

  # shellcheck disable=SC1090
  source "$ROS_SETUP"
  # shellcheck disable=SC1090
  source "$ROBOT_SETUP"
  # shellcheck disable=SC1090
  source "$NAV_SETUP"

  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-225}"
  export ROS_LOG_DIR="/root/.ros/log"
  mkdir -p "$ROS_LOG_DIR" "$STATE_DIR"
}

node_exists() {
  local wanted="$1"
  ros2 node list 2>/dev/null | grep -Fxq "$wanted"
}

wait_for_node() {
  local wanted="$1"
  local seconds="$2"
  local loops=$((seconds * 2))
  local index
  for ((index = 0; index < loops; index++)); do
    if node_exists "$wanted"; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

topic_has_publisher() {
  local topic="$1"
  ros2 topic info "$topic" 2>/dev/null | grep -Eq 'Publisher count: [1-9][0-9]*'
}

wait_for_topic_publisher() {
  local topic="$1"
  local seconds="$2"
  local loops=$((seconds * 2))
  local index
  for ((index = 0; index < loops; index++)); do
    if topic_has_publisher "$topic"; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

pid_file() {
  printf '%s/%s.pid' "$STATE_DIR" "$1"
}

session_log_dir() {
  if [[ -r "$SESSION_FILE" ]]; then
    cat "$SESSION_FILE"
  else
    printf '%s/robot_a_mapping_unknown' "$ROS_LOG_DIR"
  fi
}

managed_process_is_alive() {
  local name="$1"
  local file
  local pid
  file="$(pid_file "$name")"
  [[ -r "$file" ]] || return 1
  pid="$(cat "$file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_managed() {
  local name="$1"
  shift
  local log_dir
  local log_file
  local pid

  if managed_process_is_alive "$name"; then
    note "$name 已经由本脚本启动，跳过重复启动"
    return 0
  fi

  log_dir="$(session_log_dir)"
  mkdir -p "$log_dir"
  log_file="${log_dir}/${name}.log"

  note "启动 $name，日志：$log_file"
  setsid "$@" >>"$log_file" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$(pid_file "$name")"
  sleep 0.5

  if ! kill -0 "$pid" 2>/dev/null; then
    tail -n 40 "$log_file" >&2 || true
    die "$name 启动后立即退出"
  fi
}

stop_managed() {
  local name="$1"
  local file
  local pid
  local pgid
  local count

  file="$(pid_file "$name")"
  [[ -r "$file" ]] || return 0
  pid="$(cat "$file")"

  if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$file"
    return 0
  fi

  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  note "停止 $name（PID=$pid）"

  if [[ "$pgid" == "$pid" ]]; then
    kill -INT -- "-$pid" 2>/dev/null || true
  else
    kill -INT "$pid" 2>/dev/null || true
  fi

  for count in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$file"
      return 0
    fi
    sleep 0.25
  done

  if [[ "$pgid" == "$pid" ]]; then
    kill -TERM -- "-$pid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
  sleep 1
  rm -f "$file"
}

check_required_packages() {
  local package
  local missing=0
  for package in \
    livox_ros_driver2 \
    pointcloud_to_laserscan \
    slam_toolbox \
    topic_tools \
    tf2_ros \
    robot_a_real_navigation
  do
    if ros2 pkg prefix "$package" >/dev/null 2>&1; then
      ok "找到软件包 $package"
    else
      echo "[缺少] $package" >&2
      missing=1
    fi
  done

  if [[ "$missing" -ne 0 ]]; then
    die "缺少依赖。topic_tools 可用 apt install -y ros-humble-topic-tools 安装"
  fi
}

stop_old_mapping_conflicts() {
  local localization_pattern
  local navigation_pattern
  local pid
  local remaining

  localization_pattern='^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch localization localization\.launch\.py$'
  navigation_pattern='^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch nav2_bringup sai_real_launch\.py$'

  while read -r pid; do
    [[ -n "$pid" ]] || continue
    note "停止旧定位 launch（PID=$pid）"
    kill -INT "$pid" 2>/dev/null || true
  done < <(pgrep -f "$localization_pattern" || true)

  while read -r pid; do
    [[ -n "$pid" ]] || continue
    note "停止旧 Nav2 launch（PID=$pid）"
    kill -INT "$pid" 2>/dev/null || true
  done < <(pgrep -f "$navigation_pattern" || true)

  sleep 5
  remaining="$(ros2 node list 2>/dev/null | grep -E '^/(fast_lio|ekf_filter_node|localization_node|map_server|controller_server|planner_server)$' || true)"
  if [[ -n "$remaining" ]]; then
    echo "$remaining" >&2
    die "旧定位或 Nav2 节点仍在运行，可能被系统服务自动重启；不要继续启动建图"
  fi
  ok "旧定位和旧 Nav2 已停止"
}

livox_publishes_pointcloud2() {
  ros2 node info "$LIVOX_NODE" 2>/dev/null \
    | grep -Fq '/livox/lidar: sensor_msgs/msg/PointCloud2'
}

find_livox_launch() {
  local share
  local candidate
  share="$(ros2 pkg prefix livox_ros_driver2)/share/livox_ros_driver2"

  # MID360s is preferred because it is the verified entry on this robot.
  for candidate in rviz_MID360s_launch.py rviz_MID360_launch.py; do
    if find "$share" -type f -name "$candidate" -print -quit | grep -q .; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

ensure_livox() {
  local launch_file

  if node_exists "$LIVOX_NODE"; then
    if livox_publishes_pointcloud2; then
      ok "复用已运行的 Livox PointCloud2 驱动"
    else
      die "已有 Livox 节点未发布 sensor_msgs/msg/PointCloud2，请停止它并使用 xfer_format=0"
    fi
  else
    launch_file="$(find_livox_launch)" || die "没有找到 rviz_MID360s_launch.py 或 rviz_MID360_launch.py"
    start_managed livox_driver ros2 launch livox_ros_driver2 "$launch_file"
    wait_for_node "$LIVOX_NODE" 20 || die "等待 Livox 节点超时，请查看 $(session_log_dir)/livox_driver.log"
    livox_publishes_pointcloud2 || die "Livox 启动后没有发布 PointCloud2"
    ok "Livox 已通过 $launch_file 启动"
    note "该 Livox 入口可能额外打开一个点云 RViz；建图 RViz 出现后可关闭点云 RViz窗口"
  fi

  if ! timeout -k 1s 8s ros2 topic echo \
      /livox/lidar sensor_msgs/msg/PointCloud2 \
      --once --qos-reliability best_effort --qos-depth 1 --field header \
      >/dev/null 2>&1; then
    die "发现 /livox/lidar，但 8 秒内没有收到 PointCloud2 数据"
  fi
  ok "已收到 Livox 点云数据"
}

ensure_chassis() {
  node_exists /mecanum_drive_controller \
    || die "mecanum_drive_controller 未运行；本脚本不会重复启动底盘硬件"
  topic_has_publisher /mecanum_drive_controller/odometry \
    || die "底盘控制器没有发布里程计"
  topic_has_publisher /mecanum_drive_controller/tf_odometry \
    || die "底盘控制器没有发布 tf_odometry"
  ok "底盘控制器和轮速里程计正常"
}

start_mapping_stack() {
  local odom_tf_check
  local livox_tf_check

  start_managed wheel_tf_relay \
    ros2 run topic_tools relay \
    /mecanum_drive_controller/tf_odometry /tf \
    --ros-args -r __node:=wheel_tf_relay
  wait_for_node "$TF_RELAY_NODE" 8 || die "轮速 TF 转发节点启动失败"

  start_managed base_footprint_tf \
    ros2 run tf2_ros static_transform_publisher \
    --x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 \
    --frame-id base_link --child-frame-id base_footprint \
    --ros-args -r __node:=base_footprint_static_tf
  wait_for_node "$FOOTPRINT_TF_NODE" 8 || die "base_footprint 静态 TF 启动失败"

  start_managed livox_to_scan \
    ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node --ros-args \
    -r __node:=livox_to_scan \
    -r cloud_in:=/livox/lidar \
    -r scan:=/scan \
    -p min_height:=-0.22 \
    -p max_height:=0.23 \
    -p range_min:=0.30 \
    -p range_max:=30.0 \
    -p angle_min:=-3.14159 \
    -p angle_max:=3.14159 \
    -p angle_increment:=0.00873
  wait_for_node "$SCAN_NODE" 8 || die "点云转 /scan 节点启动失败"

  start_managed slam_mapping \
    ros2 launch robot_a_real_navigation real_robot_mapping.launch.py \
    start_hardware:=false use_rviz:=true

  wait_for_node "$SLAM_NODE" 15 || die "SLAM Toolbox 启动失败，请查看 $(session_log_dir)/slam_mapping.log"
  wait_for_topic_publisher /scan 10 || die "/scan 没有发布者"
  wait_for_topic_publisher /map 20 || die "SLAM 已启动，但 /map 20 秒内没有发布者"

  odom_tf_check="$(session_log_dir)/odom_to_base_link.check"
  timeout -k 1s 5s ros2 run tf2_ros tf2_echo odom base_link \
    >"$odom_tf_check" 2>&1 || true
  if ! grep -q 'Translation:' "$odom_tf_check"; then
    die "没有得到 odom -> base_link，请检查 wheel_tf_relay"
  fi

  livox_tf_check="$(session_log_dir)/base_link_to_livox.check"
  timeout -k 1s 5s ros2 run tf2_ros tf2_echo base_link livox_frame \
    >"$livox_tf_check" 2>&1 || true
  if ! grep -q 'Translation:' "$livox_tf_check"; then
    die "没有得到 base_link -> livox_frame，请检查 robot_state_publisher"
  fi

  ok "建图链路启动完成"
}

send_zero_velocity() {
  timeout -k 1s 3s ros2 topic pub --once \
    /mecanum_drive_controller/reference_unstamped \
    geometry_msgs/msg/Twist \
    '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' \
    >/dev/null 2>&1 || true
}

show_status() {
  echo "===== RobotA 建图状态 ====="
  local item
  for item in \
    "$LIVOX_NODE" \
    /mecanum_drive_controller \
    "$TF_RELAY_NODE" \
    "$SCAN_NODE" \
    "$FOOTPRINT_TF_NODE" \
    "$SLAM_NODE"
  do
    if node_exists "$item"; then
      echo "[运行] $item"
    else
      echo "[缺少] $item"
    fi
  done

  for item in /livox/lidar /scan /map; do
    if topic_has_publisher "$item"; then
      echo "[发布] $item"
    else
      echo "[无发布] $item"
    fi
  done

  if [[ -r "$SESSION_FILE" ]]; then
    echo "日志目录：$(session_log_dir)"
  fi
}

start_action() {
  require_root
  load_ros_environment

  if node_exists "$SLAM_NODE"; then
    note "SLAM Toolbox 已经运行，不重复启动"
    show_status
    return 0
  fi

  START_IN_PROGRESS=1
  START_COMPLETED=0

  local log_dir
  log_dir="${ROS_LOG_DIR}/robot_a_mapping_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$log_dir"
  printf '%s\n' "$log_dir" >"$SESSION_FILE"

  check_required_packages
  stop_old_mapping_conflicts
  ensure_chassis
  ensure_livox
  start_mapping_stack
  START_COMPLETED=1
  show_status

  cat <<'EOF'

===== 下一步 =====
1. 在建图 RViz 中确认 Fixed Frame 是 map。
2. 先低速原地旋转约一圈，再沿场地边缘缓慢行驶并回到起点。
3. 不要只原地旋转；只有位置发生变化才能建立完整房间和通道。
4. 保存地图：sudo bash robot_a_mapping.sh save robot_a_map
5. 停止建图：sudo bash robot_a_mapping.sh stop
EOF
}

cleanup_failed_start() {
  local exit_code=$?
  if [[ "$START_IN_PROGRESS" -eq 1 && "$START_COMPLETED" -eq 0 ]]; then
    echo "[清理] 启动未完成，正在停止本次已经启动的进程……" >&2
    send_zero_velocity || true
    stop_managed slam_mapping || true
    stop_managed livox_to_scan || true
    stop_managed base_footprint_tf || true
    stop_managed wheel_tf_relay || true
    stop_managed livox_driver || true
  fi
  return "$exit_code"
}

save_action() {
  require_root
  load_ros_environment
  [[ "$MAP_NAME" =~ ^[A-Za-z0-9_-]+$ ]] \
    || die "地图名只能包含英文字母、数字、下划线和短横线"
  topic_has_publisher /map || die "/map 没有发布者，当前不能保存地图"

  mkdir -p "$MAP_DIR"
  note "保存地图到 ${MAP_DIR}/${MAP_NAME}.yaml"
  ros2 run nav2_map_server map_saver_cli -f "${MAP_DIR}/${MAP_NAME}"

  [[ -f "${MAP_DIR}/${MAP_NAME}.yaml" ]] || die "没有生成地图 YAML"
  chown -R niic:niic "$MAP_DIR"
  ok "地图保存完成：${MAP_DIR}/${MAP_NAME}.yaml"
}

stop_action() {
  require_root
  load_ros_environment
  send_zero_velocity
  stop_managed slam_mapping
  stop_managed livox_to_scan
  stop_managed base_footprint_tf
  stop_managed wheel_tf_relay
  stop_managed livox_driver
  rm -f "$SESSION_FILE"
  ok "本脚本启动的建图进程已经停止，底盘已发送零速度"
  note "旧定位和 Nav2 不会自动恢复；导航时请使用保存的地图重新启动 Nav2"
}

trap cleanup_failed_start EXIT

case "$ACTION" in
  start)
    start_action
    ;;
  status)
    require_root
    load_ros_environment
    show_status
    ;;
  save)
    save_action
    ;;
  stop)
    stop_action
    ;;
  restart)
    stop_action
    start_action
    ;;
  *)
    cat >&2 <<EOF
用法：
  sudo bash $0 start
  sudo bash $0 status
  sudo bash $0 save 地图名
  sudo bash $0 stop
  sudo bash $0 restart
EOF
    exit 2
    ;;
esac
