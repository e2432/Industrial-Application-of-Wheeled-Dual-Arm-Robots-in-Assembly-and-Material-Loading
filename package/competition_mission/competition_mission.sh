#!/usr/bin/env bash
# competition_mission：四组抓取/放置循环及回到 start 的一键工作流
#
# 运行方式：先执行 sudo -i 进入 root，再运行本脚本。
# 本版本只允许 root 运行；脚本内部不读取密码，也不再调用 sudo。

set -Eeuo pipefail

# ======================== 现场配置 ========================
ROS_ENV="/opt/robot/current/bin/ros2env"
ROBOT_WS="/home/niic/robot_a_ws/install/setup.bash"
BOX_DIR="/home/niic/Desktop/competition/vision/box"
MISSION_NODE="$BOX_DIR/competition_mission.py"
AUTO_LOCALIZATION_SCRIPT="/home/niic/Desktop/onemission/auto_global_localization.py"
WAYPOINT_FILE="/home/niic/robot_a_ws/src/robot_a_real_navigation/config/competition_waypoints.yaml"
# 部署前必须替换为原机地图 YAML 的绝对路径。
MAP_FILE="__MAP_FILE__"
RVIZ_CONFIG="/home/niic/robot_a_ws/install/robot_a_real_navigation/share/robot_a_real_navigation/rviz/robot_a_real_navigation.rviz"
BOX_POSE_ACTIVATE="/root/activate_box_pose.sh"

NAV_CMD_VEL_TOPIC="/cmd_vel"
NAV_ARBITER_INPUT_TOPIC="/box_mission/nav_cmd_vel"
MANUAL_CMD_VEL_TOPIC="/box_mission/manual_cmd_vel"
BASE_CMD_VEL_TOPIC="/mecanum_drive_controller/reference_unstamped"

VISION_SCRIPT=""
GRASP_SCRIPT=""
LIFT_SCRIPT=""
RELEASE_SCRIPT=""
RESET_SCRIPT=""

GRASP_WAYPOINT_ID="grab1"
LIFT_WAYPOINT_ID="grab1"
RELEASE_WAYPOINT_ID="drop1"
FINAL_WAYPOINT_ID="start"
RESET_WAYPOINT_ID="start"

TAB_START_INTERVAL=0.5
TAB_START_TIMEOUT=15
DEPENDENCY_TIMEOUT=420

# Nav2 返回 ABORTED 时，任务节点会再用 map -> base_link 的真实位姿确认。
# 现场可按停车精度调整：位置单位 m，角度单位 deg。
ARRIVAL_POSITION_TOLERANCE=0.20
ARRIVAL_YAW_TOLERANCE_DEG=20.0
# 抓取、抬升和放置点使用更严格的到位阈值。
TASK_ARRIVAL_POSITION_TOLERANCE=0.08
TASK_ARRIVAL_YAW_TOLERANCE_DEG=8.0
# The grasp waypoint is stricter than lift/drop.  Nav2 success alone is not
# enough; box_waypoint_mission.py performs a mandatory final refinement.
GRASP_ARRIVAL_POSITION_TOLERANCE=0.03
GRASP_ARRIVAL_YAW_TOLERANCE_DEG=2.0
ARRIVAL_CHECK_TIMEOUT=3
NAV_RETRIES=4
NAV_LIFECYCLE_TIMEOUT=90
TF_READY_TIMEOUT=90
LOCAL_COSTMAP_CLEAR_SERVICE="/local_costmap/clear_entirely_local_costmap"
GLOBAL_COSTMAP_CLEAR_SERVICE="/global_costmap/clear_entirely_global_costmap"
COSTMAP_SERVICE_TIMEOUT=4.0
COSTMAP_REPOPULATE_DELAY=1.5
FINE_ALIGNMENT_START_RADIUS=0.25
FINE_ALIGNMENT_TIMEOUT=30
FINE_ALIGNMENT_MAX_LINEAR_SPEED=0.025
FINE_ALIGNMENT_MAX_ANGULAR_SPEED=0.06
FINE_ALIGNMENT_MIN_ANGULAR_SPEED=0.025

# 软件 OpenGL 能规避部分 Jetson/root RViz 黑色画布问题；若现场软件
# 渲染不可用，RViz 标签页会自动回退到 NVIDIA OpenGL。
RVIZ_USE_SOFTWARE_OPENGL=1

# ---------------- 视觉识别纠错（不调用 Nav2） ----------------
VISION_DETECTION_TIMEOUT=10.0
VISION_STABLE_SAMPLES=1
VISION_SAMPLE_MAX_AGE=2.0
VISION_POINT_SYNC_TOLERANCE=0.20
VISION_RECOVERY_CYCLES=2
VISION_RESTART_DELAY=1.0
VISION_SCAN_ANGLE_DEG=8.0
VISION_SCAN_SPEED=0.04
VISION_SCAN_MIN_SPEED=0.025
VISION_SCAN_YAW_TOLERANCE_DEG=1.0
VISION_SCAN_MOVE_TIMEOUT=8.0
VISION_SCAN_DWELL=3.0
VISION_SCAN_MAX_POSITION_DRIFT=0.04
VISION_SCAN_TRANSLATION_DISTANCE=0.06
VISION_SCAN_LINEAR_SPEED=0.025
VISION_SCAN_MIN_LINEAR_SPEED=0.012
VISION_SCAN_POSITION_TOLERANCE=0.01
VISION_SCAN_LATERAL_DRIFT=0.03
VISION_SCAN_TRANSLATION_TIMEOUT=8.0
# IK-driven chassis recovery.  0.50m is a hard result constraint, not a
# commanded movement distance.
BASE_RECOVERY_MIN_RESULT_X=0.50
BASE_RECOVERY_RESULT_MARGIN=0.02
BASE_RECOVERY_SEARCH_STEP=0.02
BASE_RECOVERY_MAX_TRANSLATION=0.20
BASE_RECOVERY_MAX_YAW_DEG=20.0
BASE_RECOVERY_CYCLES=3
BASE_RECOVERY_POSITION_TOLERANCE=0.01
BASE_RECOVERY_TIMEOUT=20.0
BASE_RECOVERY_TF_STABLE_SAMPLES=5
BASE_RECOVERY_TF_STABLE_POSITION=0.005
BASE_RECOVERY_TF_STABLE_YAW_DEG=0.5
BASE_RECOVERY_TF_STABLE_TIMEOUT=6.0
# =========================================================

if (( EUID != 0 )); then
  echo "错误：本版本必须直接在 root 用户下运行。" >&2
  echo "请先执行：sudo -i" >&2
  echo "然后执行：cd /home/niic/Desktop && ./competition_mission.sh" >&2
  exit 1
fi

if [[ "$MAP_FILE" == "__MAP_FILE__" ]]; then
  echo "错误：请先把 competition_mission.sh 中的 MAP_FILE 占位符替换为地图 YAML 绝对路径。" >&2
  exit 1
fi

for command_name in gnome-terminal; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：系统缺少命令：$command_name" >&2
    exit 1
  }
done

for file in \
  "$ROS_ENV" \
  "$ROBOT_WS" \
  "$MISSION_NODE" \
  "$BOX_POSE_ACTIVATE" \
  "$WAYPOINT_FILE" \
  "$MAP_FILE"; do
  [[ -r "$file" ]] || {
    echo "错误：找不到或无法读取：$file" >&2
    exit 1
  }
done

# 在机器人开始移动前确认视觉环境确实切换到了 NumPy 1.x。
if ! vision_numpy_version=$(
  bash -lc "source '$BOX_POSE_ACTIVATE' && python3 -c 'import numpy; print(numpy.__version__)'" \
    | tail -n 1
); then
  echo "错误：加载视觉环境失败：$BOX_POSE_ACTIVATE" >&2
  exit 1
fi
vision_numpy_major=${vision_numpy_version%%.*}
if [[ ! "$vision_numpy_major" =~ ^[0-9]+$ ]] \
   || (( vision_numpy_major >= 2 )); then
  echo "错误：视觉环境中的 NumPy 版本仍为 $vision_numpy_version，需要 NumPy 1.x 才能兼容 cv_bridge。" >&2
  exit 1
fi
echo "视觉环境检查通过：NumPy $vision_numpy_version"

# 如果 sudo -i 没有保留图形变量，补上机器人桌面的常用值。
export DISPLAY="${DISPLAY:-:0}"
if [[ -z "${XAUTHORITY:-}" && -r /home/niic/.Xauthority ]]; then
  export XAUTHORITY=/home/niic/.Xauthority
fi

# 在启动 root D-Bus/GNOME Terminal 之前，先使用真正属于 root 的目录，
# 修复“/run/user/1000 is owned by uid 1000, not our uid 0”。
export XDG_RUNTIME_DIR=/tmp/mission-box-runtime-root
mkdir -p "$XDG_RUNTIME_DIR"
chmod 0700 "$XDG_RUNTIME_DIR"

# 第一次进入时创建独立 root D-Bus 会话；重入后才继续启动标签页。
ROOT_DBUS_STAGE="--root-dbus-stage"
if [[ "${1:-}" != "$ROOT_DBUS_STAGE" ]] \
   && command -v dbus-run-session >/dev/null 2>&1; then
  SCRIPT_PATH=$(readlink -f "$0")
  unset DBUS_SESSION_BUS_ADDRESS
  exec dbus-run-session -- bash "$SCRIPT_PATH" "$ROOT_DBUS_STAGE"
fi

echo "当前工作流用户：$(id -un)，EUID=$EUID"
echo "脚本内部不会读取密码，也不会执行 sudo。"

# 用户要求放在最前面的权限修复。当前已经是 root。
chown niic:niic "$WAYPOINT_FILE"
chmod 0644 "$WAYPOINT_FILE"
echo "路点文件权限已修复：niic:niic，0644"

resolve_unique_script() {
  local label=$1 configured=$2 pattern=$3
  local -a matches=()

  if [[ -n "$configured" ]]; then
    [[ -r "$configured" ]] || {
      echo "错误：$label 文件不存在或不可读：$configured" >&2
      return 1
    }
    printf '%s\n' "$configured"
    return 0
  fi

  while IFS= read -r candidate; do
    matches+=("$candidate")
  done < <(
    find "$BOX_DIR" -maxdepth 2 -type f -name "$pattern" -print 2>/dev/null \
      | sort
  )

  if (( ${#matches[@]} == 1 )); then
    printf '%s\n' "${matches[0]}"
    return 0
  fi

  if (( ${#matches[@]} == 0 )); then
    echo "错误：没有找到 $label，匹配规则：$pattern" >&2
  else
    echo "错误：找到多个 $label，请在脚本顶部填写完整路径：" >&2
    printf '  %s\n' "${matches[@]}" >&2
  fi
  return 1
}

VISION_SCRIPT=$(resolve_unique_script \
  "视觉脚本" "$VISION_SCRIPT" "box_ros2_robust.py") || exit 1
GRASP_SCRIPT=$(resolve_unique_script \
  "抓取脚本" "$GRASP_SCRIPT" \
  "bridge_xyz_offset_box_wide_descend_postlift.py") || exit 1
LIFT_SCRIPT=$(resolve_unique_script \
  "抬升脚本" "$LIFT_SCRIPT" \
  "box_dual_arm_place_height_constrained_v4_smooth.py") || exit 1
RELEASE_SCRIPT=$(resolve_unique_script \
  "释放脚本" "$RELEASE_SCRIPT" \
  "box_dual_arm_release_outward_3cm.py") || exit 1
RESET_SCRIPT=$(resolve_unique_script \
  "关节归零脚本" "$RESET_SCRIPT" \
  "box_all_joints_zero.py") || exit 1

if [[ -z "$AUTO_LOCALIZATION_SCRIPT" \
      && -r /home/niic/Desktop/auto_global_localization.py ]]; then
  AUTO_LOCALIZATION_SCRIPT="/home/niic/Desktop/auto_global_localization.py"
else
  AUTO_LOCALIZATION_SCRIPT=$(resolve_unique_script \
    "自动定位脚本" "$AUTO_LOCALIZATION_SCRIPT" \
    "auto_global_localization.py") || exit 1
fi

echo "自动定位脚本：$AUTO_LOCALIZATION_SCRIPT"

VISION_COMMAND="python3 $VISION_SCRIPT --ros-args -p target_center_band_ratio:=0.25 -p target_depth_roi_scale:=0.50 -p target_depth_percentile:=30.0"
GRASP_COMMAND="python3 $GRASP_SCRIPT --ros-args -p enable_motion:=true -p force_execute_confirmation:=false -p require_execute_confirmation:=false -p velocity_scale:=0.30 -p post_grasp_lift_m:=0.10 -p base_recovery_enabled:=true -p base_recovery_min_result_x_m:=$BASE_RECOVERY_MIN_RESULT_X -p base_recovery_result_margin_m:=$BASE_RECOVERY_RESULT_MARGIN -p base_recovery_search_step_m:=$BASE_RECOVERY_SEARCH_STEP -p base_recovery_max_translation_m:=$BASE_RECOVERY_MAX_TRANSLATION -p base_recovery_max_yaw_deg:=$BASE_RECOVERY_MAX_YAW_DEG"
LIFT_COMMAND="python3 $LIFT_SCRIPT -z 0.7 --velocity-scale 0.50 --execute"
RELEASE_COMMAND="python3 $RELEASE_SCRIPT --execute"
RESET_COMMAND="python3 $RESET_SCRIPT --execute --duration 10.0 --steps 100"

# ---------------- 多个独立 root 标签页 ----------------
tab_script_dir=$(mktemp -d /tmp/competition_mission_tabs.XXXXXX) || exit 1
chmod 0700 "$tab_script_dir"

started_dir="$tab_script_dir/started"
mkdir -m 0700 "$started_dir"

nav_status="$tab_script_dir/nav.status"
speed_status="$tab_script_dir/speed.status"
localization_status="$tab_script_dir/localization.status"
rviz_status="$tab_script_dir/rviz.status"

first_tab=1
add_root_tab() {
  local title=$1
  local command_body=$2
  local tab_number=${title%% *}
  local tab_script="$tab_script_dir/tab_${tab_number}.sh"
  local started_file="$started_dir/${tab_number}.started"
  local root_payload wrapper root_command

  root_payload='if (( EUID != 0 )); then echo "错误：此标签页不是真正的 root，EUID=$EUID"; exit 1; fi'$'\n'
  root_payload+='rm -f -- "$0"'$'\n'
  root_payload+="printf '%s\\n' started > $(printf '%q' "$started_file")"$'\n'
  root_payload+="$command_body"$'\n\n'
  root_payload+='tab_status=$?'$'\n'
  root_payload+='echo'$'\n'
  root_payload+="echo \"===== ${title} 已结束，退出状态：\${tab_status} =====\""$'\n'
  root_payload+='echo "当前保留 root 命令行；输入 exit 可关闭此标签页。"'$'\n'
  root_payload+='exec bash -i'$'\n'

  printf '%s\n' "$root_payload" > "$tab_script"
  chmod 0700 "$tab_script"

  printf -v root_command 'exec bash %q' "$tab_script"
  printf -v wrapper '/usr/bin/env HOME=/root USER=root LOGNAME=root bash -ic %q' \
    "$root_command"

  if (( first_tab == 1 )); then
    gnome-terminal --window --maximize --title="$title" --command="$wrapper"
    first_tab=0
  else
    gnome-terminal --tab --title="$title" --command="$wrapper"
  fi

  for ((wait_step = 0; wait_step < TAB_START_TIMEOUT * 10; wait_step++)); do
    [[ -f "$started_file" ]] && break
    sleep 0.1
  done

  if [[ ! -f "$started_file" ]]; then
    echo "错误：$title 未能在 ${TAB_START_TIMEOUT}s 内以 root 启动。" >&2
    return 1
  fi

  echo "已启动：$title"
  sleep "$TAB_START_INTERVAL"
}

# ---------------- 每个 root 标签页执行的命令 ----------------
read -r -d '' NAV2_CMD <<'EOF' || true
set +u
source '__ROS_ENV__' || exit 1
source '__ROBOT_WS__' || exit 1
set -u
export ROS_LOG_DIR=/root/.ros/log
mkdir -p "$ROS_LOG_DIR"

# Nav2 launch 内包含 RViz。先使用 NVIDIA 后端启动；定位结束后第 04 页
# 会关闭这个早期 RViz，再以干净队列重新打开。
export QT_X11_NO_MITSHM=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
unset LIBGL_ALWAYS_SOFTWARE

rm -f -- '__NAV_STATUS__'
ros2 launch robot_a_real_navigation real_robot_navigation.launch.py \
  map:='__MAP_FILE__' \
  start_hardware:=false &
nav_pid=$!

nav_ready=0
action_ready_streak=0
nav_action_name=""
bt_node_name=""
bt_state="unavailable"
for ((i = 0; i < 180; i++)); do
  if ! kill -0 "$nav_pid" 2>/dev/null; then
    break
  fi

  action_ready=0
  lifecycle_active=0
  nav_action_name=$(timeout 3 ros2 action list 2>/dev/null \
    | grep -E '(^|/)(navigate_to_pose|navigate_through_poses)$' \
    | head -n 1 || true)
  [[ -n "$nav_action_name" ]] && action_ready=1

  bt_node_name=$(timeout 3 ros2 node list 2>/dev/null \
    | grep -E '(^|/)bt_navigator$' \
    | head -n 1 || true)
  if [[ -n "$bt_node_name" ]]; then
    bt_state=$(timeout 3 ros2 lifecycle get "$bt_node_name" 2>/dev/null || true)
    printf '%s\n' "$bt_state" | grep -Eqi '(^|[[:space:]])active([[:space:]\[]|$)' \
      && lifecycle_active=1
  else
    bt_state="bt_navigator node not discovered yet"
  fi

  if (( action_ready == 1 && lifecycle_active == 1 )); then
    nav_ready=1
    break
  fi

  # Vendor Nav2 builds can expose a usable action server even when querying the
  # lifecycle state from another root terminal is unreliable.  Three
  # consecutive discoveries are enough to release the relay and localization
  # tabs.  The mission node still checks lifecycle and TF before its first goal.
  if (( action_ready == 1 )); then
    ((action_ready_streak += 1))
  else
    action_ready_streak=0
  fi
  if (( action_ready_streak >= 3 )); then
    echo "Nav2 action has been visible for three consecutive checks; continuing even though lifecycle CLI did not report active."
    nav_ready=1
    break
  fi

  if (( i % 10 == 0 )); then
    echo "Waiting for Nav2: action=${nav_action_name:-missing}, bt_node=${bt_node_name:-missing}, bt_state=${bt_state:-unknown}"
  fi

  sleep 1
done

if (( nav_ready == 0 )); then
  printf '%s\n' failed > '__NAV_STATUS__'
  echo "错误：Nav2 在 180 秒内没有出现可用导航 action。"
  echo "最后导航 action：${nav_action_name:-missing}"
  echo "最后 bt_navigator 节点：${bt_node_name:-missing}"
  echo "最后 bt_navigator 状态：${bt_state:-unknown}"
  kill -INT "$nav_pid" 2>/dev/null || true
  wait "$nav_pid" 2>/dev/null || true
  exit 1
fi

printf '%s\n' success > '__NAV_STATUS__'
echo "Nav2 启动接口已就绪：action=${nav_action_name:-unknown}, bt_node=${bt_node_name:-unknown}, bt_state=${bt_state:-unknown}。"
wait "$nav_pid"
EOF

read -r -d '' SPEED_CMD <<'EOF' || true
set +u
source '__ROS_ENV__' || exit 1
source '__ROBOT_WS__' || exit 1
set -u
export ROS_LOG_DIR=/root/.ros/log
mkdir -p "$ROS_LOG_DIR"

echo "等待 Nav2 就绪……"
for ((i = 0; i < __DEPENDENCY_TIMEOUT__; i++)); do
  [[ -f '__NAV_STATUS__' ]] && break
  if (( i > 0 && i % 15 == 0 )); then
    echo "仍在等待 01 标签页写入 Nav2 就绪状态；请查看 01 标签页中的 action/bt_state 诊断。"
  fi
  sleep 1
done
[[ -f '__NAV_STATUS__' ]] || { echo "错误：等待 Nav2 超时。"; exit 1; }
[[ $(<'__NAV_STATUS__') == success ]] || { echo "错误：Nav2 启动失败。"; exit 1; }

rm -f -- '__SPEED_STATUS__'
ros2 run topic_tools relay \
  '__NAV_CMD_VEL_TOPIC__' \
  '__NAV_ARBITER_INPUT_TOPIC__' \
  --ros-args -r __node:=nav2_cmd_vel_relay &
relay_pid=$!
sleep 3

if ! kill -0 "$relay_pid" 2>/dev/null; then
  printf '%s\n' failed > '__SPEED_STATUS__'
  echo "错误：速度转发进程提前退出。"
  wait "$relay_pid" 2>/dev/null || true
  exit 1
fi

printf '%s\n' success > '__SPEED_STATUS__'
echo "Nav2 速度已转发到任务节点仲裁输入，底盘只接受仲裁后的唯一输出。"
wait "$relay_pid"
EOF

read -r -d '' LOCALIZATION_CMD <<'EOF' || true
set +u
source '__ROS_ENV__' || exit 1
source '__ROBOT_WS__' || exit 1
set -u
export ROS_LOG_DIR=/root/.ros/log
mkdir -p "$ROS_LOG_DIR"

echo "等待 Nav2 启动接口和速度转发就绪……"
for dependency in '__NAV_STATUS__' '__SPEED_STATUS__'; do
  for ((i = 0; i < __DEPENDENCY_TIMEOUT__; i++)); do
    [[ -f "$dependency" ]] && break
    if (( i > 0 && i % 15 == 0 )); then
      echo "仍在等待：$dependency"
    fi
    sleep 1
  done
  [[ -f "$dependency" ]] || { echo "错误：等待依赖超时：$dependency"; exit 1; }
  [[ $(<"$dependency") == success ]] || { echo "错误：前置步骤失败：$dependency"; exit 1; }
done

rm -f -- '__LOCALIZATION_STATUS__'
echo "依赖已就绪，现在启动自动定位；此时 RViz 的 Localization 才会由 inactive 转为 active。"
python3 '__AUTO_LOCALIZATION_SCRIPT__'
localization_rc=$?

if (( localization_rc == 0 )); then
  printf '%s\n' success > '__LOCALIZATION_STATUS__'
  echo "自动定位成功并已自动退出。"
else
  printf '%s\n' failed > '__LOCALIZATION_STATUS__'
  echo "错误：自动定位失败，返回码：$localization_rc"
  exit "$localization_rc"
fi
EOF

read -r -d '' RVIZ_CMD <<'EOF' || true
set +u
source '__ROS_ENV__' || exit 1
source '__ROBOT_WS__' || exit 1
set -u
export ROS_LOG_DIR=/root/.ros/log
mkdir -p "$ROS_LOG_DIR"

for ((i = 0; i < __DEPENDENCY_TIMEOUT__; i++)); do
  [[ -f '__LOCALIZATION_STATUS__' ]] && break
  sleep 1
done
[[ -f '__LOCALIZATION_STATUS__' ]] || { echo "错误：等待自动定位超时。"; exit 1; }
[[ $(<'__LOCALIZATION_STATUS__') == success ]] || { echo "错误：自动定位失败，不刷新 RViz。"; exit 1; }

echo "关闭启动过早的旧 RViz，清空旧 TF/雷达消息队列……"
pkill -TERM -x rviz2 2>/dev/null || true
sleep 3
pkill -KILL -x rviz2 2>/dev/null || true
sleep 2

rm -f -- '__RVIZ_STATUS__'
export QT_X11_NO_MITSHM=1
export LIBGL_DRI3_DISABLE=1
export OGRE_RTT_MODE=Copy

start_rviz() {
  local mode=$1
  if [[ "$mode" == software ]]; then
    export LIBGL_ALWAYS_SOFTWARE=1
    export MESA_GL_VERSION_OVERRIDE=3.3
    unset __GLX_VENDOR_LIBRARY_NAME
  else
    unset LIBGL_ALWAYS_SOFTWARE
    unset MESA_GL_VERSION_OVERRIDE
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
  fi

  rviz2 -d '__RVIZ_CONFIG__' --ros-args -p use_sim_time:=false &
  rviz_pid=$!
  sleep 6
  kill -0 "$rviz_pid" 2>/dev/null
}

if [[ '__RVIZ_USE_SOFTWARE_OPENGL__' == 1 ]]; then
  echo "以软件 OpenGL 启动刷新后的 root RViz……"
  if ! start_rviz software; then
    echo "软件 OpenGL 启动失败，回退 NVIDIA OpenGL。"
    start_rviz nvidia || { printf '%s\n' failed > '__RVIZ_STATUS__'; exit 1; }
  fi
else
  start_rviz nvidia || { printf '%s\n' failed > '__RVIZ_STATUS__'; exit 1; }
fi

printf '%s\n' success > '__RVIZ_STATUS__'
echo "刷新后的 RViz 已启动，PID=$rviz_pid。"
wait "$rviz_pid"
EOF

read -r -d '' BOX_MISSION_CMD <<'EOF' || true
set +u
source '__ROS_ENV__' || exit 1
source '__ROBOT_WS__' || exit 1
set -u
export ROS_LOG_DIR=/root/.ros/log
mkdir -p "$ROS_LOG_DIR"

for dependency in '__NAV_STATUS__' '__LOCALIZATION_STATUS__' '__RVIZ_STATUS__'; do
  for ((i = 0; i < __DEPENDENCY_TIMEOUT__; i++)); do
    [[ -f "$dependency" ]] && break
    sleep 1
  done
  [[ -f "$dependency" ]] || { echo "错误：等待依赖超时：$dependency"; exit 1; }
  [[ $(<"$dependency") == success ]] || { echo "错误：前置步骤失败：$dependency"; exit 1; }
done

export BOX_MISSION_VISION_CMD='__VISION_COMMAND__'
export BOX_MISSION_NODE_NAME='competition_mission'
export BOX_MISSION_VISION_SETUP='__BOX_POSE_ACTIVATE__'
export BOX_MISSION_GRASP_CMD='__GRASP_COMMAND__'
export BOX_MISSION_LIFT_CMD='__LIFT_COMMAND__'
export BOX_MISSION_RELEASE_CMD='__RELEASE_COMMAND__'
export BOX_MISSION_RESET_CMD='__RESET_COMMAND__'
export BOX_MISSION_NAV_CMD_VEL_INPUT_TOPIC='__NAV_ARBITER_INPUT_TOPIC__'
export BOX_MISSION_MANUAL_CMD_VEL_TOPIC='__MANUAL_CMD_VEL_TOPIC__'
export BOX_MISSION_BASE_CMD_VEL_OUTPUT_TOPIC='__BASE_CMD_VEL_TOPIC__'
export BOX_MISSION_VISION_DETECTION_TIMEOUT='__VISION_DETECTION_TIMEOUT__'
export BOX_MISSION_VISION_STABLE_SAMPLES='__VISION_STABLE_SAMPLES__'
export BOX_MISSION_VISION_SAMPLE_MAX_AGE='__VISION_SAMPLE_MAX_AGE__'
export BOX_MISSION_VISION_POINT_SYNC_TOLERANCE='__VISION_POINT_SYNC_TOLERANCE__'
export BOX_MISSION_VISION_RECOVERY_CYCLES='__VISION_RECOVERY_CYCLES__'
export BOX_MISSION_VISION_RESTART_DELAY='__VISION_RESTART_DELAY__'
export BOX_MISSION_VISION_SCAN_ANGLE_DEG='__VISION_SCAN_ANGLE_DEG__'
export BOX_MISSION_VISION_SCAN_SPEED='__VISION_SCAN_SPEED__'
export BOX_MISSION_VISION_SCAN_MIN_SPEED='__VISION_SCAN_MIN_SPEED__'
export BOX_MISSION_VISION_SCAN_YAW_TOLERANCE_DEG='__VISION_SCAN_YAW_TOLERANCE_DEG__'
export BOX_MISSION_VISION_SCAN_MOVE_TIMEOUT='__VISION_SCAN_MOVE_TIMEOUT__'
export BOX_MISSION_VISION_SCAN_DWELL='__VISION_SCAN_DWELL__'
export BOX_MISSION_VISION_SCAN_MAX_POSITION_DRIFT='__VISION_SCAN_MAX_POSITION_DRIFT__'
export BOX_MISSION_VISION_SCAN_TRANSLATION_DISTANCE='__VISION_SCAN_TRANSLATION_DISTANCE__'
export BOX_MISSION_VISION_SCAN_LINEAR_SPEED='__VISION_SCAN_LINEAR_SPEED__'
export BOX_MISSION_VISION_SCAN_MIN_LINEAR_SPEED='__VISION_SCAN_MIN_LINEAR_SPEED__'
export BOX_MISSION_VISION_SCAN_POSITION_TOLERANCE='__VISION_SCAN_POSITION_TOLERANCE__'
export BOX_MISSION_VISION_SCAN_LATERAL_DRIFT='__VISION_SCAN_LATERAL_DRIFT__'
export BOX_MISSION_VISION_SCAN_TRANSLATION_TIMEOUT='__VISION_SCAN_TRANSLATION_TIMEOUT__'
export BOX_MISSION_BASE_RECOVERY_CYCLES='__BASE_RECOVERY_CYCLES__'
export BOX_MISSION_BASE_RECOVERY_POSITION_TOLERANCE='__BASE_RECOVERY_POSITION_TOLERANCE__'
export BOX_MISSION_BASE_RECOVERY_TIMEOUT='__BASE_RECOVERY_TIMEOUT__'
export BOX_MISSION_BASE_RECOVERY_TF_STABLE_SAMPLES='__BASE_RECOVERY_TF_STABLE_SAMPLES__'
export BOX_MISSION_BASE_RECOVERY_TF_STABLE_POSITION='__BASE_RECOVERY_TF_STABLE_POSITION__'
export BOX_MISSION_BASE_RECOVERY_TF_STABLE_YAW_DEG='__BASE_RECOVERY_TF_STABLE_YAW_DEG__'
export BOX_MISSION_BASE_RECOVERY_TF_STABLE_TIMEOUT='__BASE_RECOVERY_TF_STABLE_TIMEOUT__'

echo "所有依赖已就绪，开始箱体多点运输任务。"
python3 '__MISSION_NODE__' \
  --waypoints '__WAYPOINT_FILE__' \
  --box-dir '__BOX_DIR__' \
  --grasp-waypoint '__GRASP_WAYPOINT_ID__' \
  --lift-waypoint '__LIFT_WAYPOINT_ID__' \
  --release-waypoint '__RELEASE_WAYPOINT_ID__' \
  --reset-waypoint '__RESET_WAYPOINT_ID__' \
  --final-waypoint '__FINAL_WAYPOINT_ID__' \
  --arrival-position-tolerance '__ARRIVAL_POSITION_TOLERANCE__' \
  --arrival-yaw-tolerance-deg '__ARRIVAL_YAW_TOLERANCE_DEG__' \
  --task-arrival-position-tolerance '__TASK_ARRIVAL_POSITION_TOLERANCE__' \
  --task-arrival-yaw-tolerance-deg '__TASK_ARRIVAL_YAW_TOLERANCE_DEG__' \
  --grasp-arrival-position-tolerance '__GRASP_ARRIVAL_POSITION_TOLERANCE__' \
  --grasp-arrival-yaw-tolerance-deg '__GRASP_ARRIVAL_YAW_TOLERANCE_DEG__' \
  --arrival-check-timeout '__ARRIVAL_CHECK_TIMEOUT__' \
  --nav-retries '__NAV_RETRIES__' \
  --nav-lifecycle-timeout '__NAV_LIFECYCLE_TIMEOUT__' \
  --tf-ready-timeout '__TF_READY_TIMEOUT__' \
  --local-costmap-clear-service '__LOCAL_COSTMAP_CLEAR_SERVICE__' \
  --global-costmap-clear-service '__GLOBAL_COSTMAP_CLEAR_SERVICE__' \
  --costmap-service-timeout '__COSTMAP_SERVICE_TIMEOUT__' \
  --costmap-repopulate-delay '__COSTMAP_REPOPULATE_DELAY__' \
  --enable-fine-alignment \
  --fine-alignment-start-radius '__FINE_ALIGNMENT_START_RADIUS__' \
  --fine-alignment-timeout '__FINE_ALIGNMENT_TIMEOUT__' \
  --fine-alignment-max-linear-speed '__FINE_ALIGNMENT_MAX_LINEAR_SPEED__' \
  --fine-alignment-max-angular-speed '__FINE_ALIGNMENT_MAX_ANGULAR_SPEED__' \
  --fine-alignment-min-angular-speed '__FINE_ALIGNMENT_MIN_ANGULAR_SPEED__'
EOF

# 现场路径不含单引号；命令字符串中的 Python 路径同样不含单引号。
NAV2_CMD=${NAV2_CMD//__ROS_ENV__/$ROS_ENV}
NAV2_CMD=${NAV2_CMD//__ROBOT_WS__/$ROBOT_WS}
NAV2_CMD=${NAV2_CMD//__NAV_STATUS__/$nav_status}
NAV2_CMD=${NAV2_CMD//__MAP_FILE__/$MAP_FILE}

SPEED_CMD=${SPEED_CMD//__ROS_ENV__/$ROS_ENV}
SPEED_CMD=${SPEED_CMD//__ROBOT_WS__/$ROBOT_WS}
SPEED_CMD=${SPEED_CMD//__DEPENDENCY_TIMEOUT__/$DEPENDENCY_TIMEOUT}
SPEED_CMD=${SPEED_CMD//__NAV_STATUS__/$nav_status}
SPEED_CMD=${SPEED_CMD//__SPEED_STATUS__/$speed_status}
SPEED_CMD=${SPEED_CMD//__NAV_CMD_VEL_TOPIC__/$NAV_CMD_VEL_TOPIC}
SPEED_CMD=${SPEED_CMD//__NAV_ARBITER_INPUT_TOPIC__/$NAV_ARBITER_INPUT_TOPIC}
SPEED_CMD=${SPEED_CMD//__BASE_CMD_VEL_TOPIC__/$BASE_CMD_VEL_TOPIC}

LOCALIZATION_CMD=${LOCALIZATION_CMD//__ROS_ENV__/$ROS_ENV}
LOCALIZATION_CMD=${LOCALIZATION_CMD//__ROBOT_WS__/$ROBOT_WS}
LOCALIZATION_CMD=${LOCALIZATION_CMD//__DEPENDENCY_TIMEOUT__/$DEPENDENCY_TIMEOUT}
LOCALIZATION_CMD=${LOCALIZATION_CMD//__NAV_STATUS__/$nav_status}
LOCALIZATION_CMD=${LOCALIZATION_CMD//__SPEED_STATUS__/$speed_status}
LOCALIZATION_CMD=${LOCALIZATION_CMD//__LOCALIZATION_STATUS__/$localization_status}
LOCALIZATION_CMD=${LOCALIZATION_CMD//__AUTO_LOCALIZATION_SCRIPT__/$AUTO_LOCALIZATION_SCRIPT}

RVIZ_CMD=${RVIZ_CMD//__ROS_ENV__/$ROS_ENV}
RVIZ_CMD=${RVIZ_CMD//__ROBOT_WS__/$ROBOT_WS}
RVIZ_CMD=${RVIZ_CMD//__DEPENDENCY_TIMEOUT__/$DEPENDENCY_TIMEOUT}
RVIZ_CMD=${RVIZ_CMD//__LOCALIZATION_STATUS__/$localization_status}
RVIZ_CMD=${RVIZ_CMD//__RVIZ_STATUS__/$rviz_status}
RVIZ_CMD=${RVIZ_CMD//__RVIZ_CONFIG__/$RVIZ_CONFIG}
RVIZ_CMD=${RVIZ_CMD//__RVIZ_USE_SOFTWARE_OPENGL__/$RVIZ_USE_SOFTWARE_OPENGL}

BOX_MISSION_CMD=${BOX_MISSION_CMD//__ROS_ENV__/$ROS_ENV}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__ROBOT_WS__/$ROBOT_WS}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__DEPENDENCY_TIMEOUT__/$DEPENDENCY_TIMEOUT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__NAV_STATUS__/$nav_status}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__LOCALIZATION_STATUS__/$localization_status}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__RVIZ_STATUS__/$rviz_status}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_COMMAND__/$VISION_COMMAND}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__BOX_POSE_ACTIVATE__/$BOX_POSE_ACTIVATE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__GRASP_COMMAND__/$GRASP_COMMAND}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__LIFT_COMMAND__/$LIFT_COMMAND}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__RELEASE_COMMAND__/$RELEASE_COMMAND}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__RESET_COMMAND__/$RESET_COMMAND}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__NAV_ARBITER_INPUT_TOPIC__/$NAV_ARBITER_INPUT_TOPIC}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__MANUAL_CMD_VEL_TOPIC__/$MANUAL_CMD_VEL_TOPIC}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__BASE_CMD_VEL_TOPIC__/$BASE_CMD_VEL_TOPIC}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__MISSION_NODE__/$MISSION_NODE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__WAYPOINT_FILE__/$WAYPOINT_FILE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__BOX_DIR__/$BOX_DIR}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_DETECTION_TIMEOUT__/$VISION_DETECTION_TIMEOUT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_STABLE_SAMPLES__/$VISION_STABLE_SAMPLES}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SAMPLE_MAX_AGE__/$VISION_SAMPLE_MAX_AGE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_POINT_SYNC_TOLERANCE__/$VISION_POINT_SYNC_TOLERANCE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_RECOVERY_CYCLES__/$VISION_RECOVERY_CYCLES}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_RESTART_DELAY__/$VISION_RESTART_DELAY}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_ANGLE_DEG__/$VISION_SCAN_ANGLE_DEG}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_SPEED__/$VISION_SCAN_SPEED}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_MIN_SPEED__/$VISION_SCAN_MIN_SPEED}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_YAW_TOLERANCE_DEG__/$VISION_SCAN_YAW_TOLERANCE_DEG}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_MOVE_TIMEOUT__/$VISION_SCAN_MOVE_TIMEOUT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_DWELL__/$VISION_SCAN_DWELL}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_MAX_POSITION_DRIFT__/$VISION_SCAN_MAX_POSITION_DRIFT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_TRANSLATION_DISTANCE__/$VISION_SCAN_TRANSLATION_DISTANCE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_LINEAR_SPEED__/$VISION_SCAN_LINEAR_SPEED}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_MIN_LINEAR_SPEED__/$VISION_SCAN_MIN_LINEAR_SPEED}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_POSITION_TOLERANCE__/$VISION_SCAN_POSITION_TOLERANCE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_LATERAL_DRIFT__/$VISION_SCAN_LATERAL_DRIFT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__VISION_SCAN_TRANSLATION_TIMEOUT__/$VISION_SCAN_TRANSLATION_TIMEOUT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__BASE_RECOVERY_CYCLES__/$BASE_RECOVERY_CYCLES}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__BASE_RECOVERY_POSITION_TOLERANCE__/$BASE_RECOVERY_POSITION_TOLERANCE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__BASE_RECOVERY_TIMEOUT__/$BASE_RECOVERY_TIMEOUT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__BASE_RECOVERY_TF_STABLE_SAMPLES__/$BASE_RECOVERY_TF_STABLE_SAMPLES}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__BASE_RECOVERY_TF_STABLE_POSITION__/$BASE_RECOVERY_TF_STABLE_POSITION}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__BASE_RECOVERY_TF_STABLE_YAW_DEG__/$BASE_RECOVERY_TF_STABLE_YAW_DEG}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__BASE_RECOVERY_TF_STABLE_TIMEOUT__/$BASE_RECOVERY_TF_STABLE_TIMEOUT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__GRASP_WAYPOINT_ID__/$GRASP_WAYPOINT_ID}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__LIFT_WAYPOINT_ID__/$LIFT_WAYPOINT_ID}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__RELEASE_WAYPOINT_ID__/$RELEASE_WAYPOINT_ID}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__RESET_WAYPOINT_ID__/$RESET_WAYPOINT_ID}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__FINAL_WAYPOINT_ID__/$FINAL_WAYPOINT_ID}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__ARRIVAL_POSITION_TOLERANCE__/$ARRIVAL_POSITION_TOLERANCE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__ARRIVAL_YAW_TOLERANCE_DEG__/$ARRIVAL_YAW_TOLERANCE_DEG}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__TASK_ARRIVAL_POSITION_TOLERANCE__/$TASK_ARRIVAL_POSITION_TOLERANCE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__TASK_ARRIVAL_YAW_TOLERANCE_DEG__/$TASK_ARRIVAL_YAW_TOLERANCE_DEG}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__GRASP_ARRIVAL_POSITION_TOLERANCE__/$GRASP_ARRIVAL_POSITION_TOLERANCE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__GRASP_ARRIVAL_YAW_TOLERANCE_DEG__/$GRASP_ARRIVAL_YAW_TOLERANCE_DEG}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__ARRIVAL_CHECK_TIMEOUT__/$ARRIVAL_CHECK_TIMEOUT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__NAV_RETRIES__/$NAV_RETRIES}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__NAV_LIFECYCLE_TIMEOUT__/$NAV_LIFECYCLE_TIMEOUT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__TF_READY_TIMEOUT__/$TF_READY_TIMEOUT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__LOCAL_COSTMAP_CLEAR_SERVICE__/$LOCAL_COSTMAP_CLEAR_SERVICE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__GLOBAL_COSTMAP_CLEAR_SERVICE__/$GLOBAL_COSTMAP_CLEAR_SERVICE}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__COSTMAP_SERVICE_TIMEOUT__/$COSTMAP_SERVICE_TIMEOUT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__COSTMAP_REPOPULATE_DELAY__/$COSTMAP_REPOPULATE_DELAY}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__FINE_ALIGNMENT_START_RADIUS__/$FINE_ALIGNMENT_START_RADIUS}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__FINE_ALIGNMENT_TIMEOUT__/$FINE_ALIGNMENT_TIMEOUT}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__FINE_ALIGNMENT_MAX_LINEAR_SPEED__/$FINE_ALIGNMENT_MAX_LINEAR_SPEED}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__FINE_ALIGNMENT_MAX_ANGULAR_SPEED__/$FINE_ALIGNMENT_MAX_ANGULAR_SPEED}
BOX_MISSION_CMD=${BOX_MISSION_CMD//__FINE_ALIGNMENT_MIN_ANGULAR_SPEED__/$FINE_ALIGNMENT_MIN_ANGULAR_SPEED}

echo
echo "开始直接创建五个 root 标签页；后续步骤会在各自标签页等待依赖。"
launch_failed=0
add_root_tab "01 Nav2和初始RViz" "$NAV2_CMD" || launch_failed=1
(( launch_failed == 0 )) && add_root_tab "02 Nav2速度接入仲裁" "$SPEED_CMD" || launch_failed=1
(( launch_failed == 0 )) && add_root_tab "03 自动定位" "$LOCALIZATION_CMD" || launch_failed=1
(( launch_failed == 0 )) && add_root_tab "04 刷新RViz" "$RVIZ_CMD" || launch_failed=1
(( launch_failed == 0 )) && add_root_tab "05 competition_mission" "$BOX_MISSION_CMD" || launch_failed=1

if (( launch_failed != 0 )); then
  echo "启动链未完成，请查看最后一个已打开的标签页。" >&2
  exit 1
fi

echo
echo "五个 root 标签页均已创建："
echo "  01 Nav2和初始RViz"
echo "  02 速度转发（等待 Nav2）"
echo "  03 自动定位（等待 Nav2 和速度转发）"
echo "  04 刷新RViz（等待自动定位成功）"
echo "  05 competition_mission（等待定位和 RViz 成功）"
echo "整个过程没有读取密码，也没有调用 sudo。"

# dbus-run-session 的父进程必须保持运行，否则 root 的 D-Bus 被关闭后，
# 新建的 GNOME Terminal 窗口也可能一起消失。这不是任务卡住。
if [[ "${1:-}" == "$ROOT_DBUS_STAGE" ]]; then
  echo
  echo "root 终端会话管理器正在保持运行，这是正常状态。"
  echo "请在各标签页观察任务；关闭全部 root 标签页后，本启动器会自动结束。"
  while pgrep -u 0 -f 'gnome-terminal-server' >/dev/null 2>&1; do
    sleep 2
  done
fi
