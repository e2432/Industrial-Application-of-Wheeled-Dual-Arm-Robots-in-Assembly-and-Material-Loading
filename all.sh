#!/usr/bin/env bash

# Robot A 一键启动脚本
# 在一个 GNOME Terminal 窗口中创建多个标签页；每个标签页中的命令均以 root 执行。

set -u

if [[ ${EUID} -eq 0 ]]; then
  echo "请以桌面登录用户运行本脚本，不要在脚本前加 sudo。"
  echo "脚本会先申请一次 sudo 权限，随后所有标签页中的命令都会以 root 执行。"
  exit 1
fi

if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "错误：未找到 gnome-terminal，无法创建同一窗口中的多个标签页。"
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "错误：未找到 sudo。"
  exit 1
fi

echo "所有标签页将共用本次 root 验证，密码不会写入文件。"
read -r -s -p "[sudo] password for ${USER}: " ROOT_PASSWORD
echo

# 先验证密码。使用 shell 内建 printf，避免把密码放到外部进程的命令行参数中。
if ! printf '%s\n' "${ROOT_PASSWORD}" | sudo -S -k -p '' -v; then
  unset ROOT_PASSWORD
  echo "错误：sudo 密码验证失败。"
  exit 1
fi

read -r -d '' ARM_CMD <<'EOF' || true
cd /home/niic/project/20260623/20260116/test_robot_A || exit 1
source /opt/robot/current/bin/ros2env
source install/setup.bash
ros2 run upper_ethercat ethercat-upper -x 6 -l 0
EOF

read -r -d '' CHASSIS_CMD <<'EOF' || true
cd /home/niic/project/20260623/20260116/test_robot_A || exit 1
source /opt/robot/current/bin/ros2env
source install/setup.bash
ros2 launch ethercat_chassis ethercat_chassis.launch.py
EOF

read -r -d '' MOVEIT_CMD <<'EOF' || true
cd /home/niic/project/20260623/20260116/test_robot_A || exit 1
source /opt/robot/current/bin/ros2env
source install/setup.bash
ros2 launch robot_a_moveit_config demo.launch.py
EOF

read -r -d '' CAMERA_CMD <<'EOF' || true
cd /home/niic/project/20260623/20260116/test_robot_A || exit 1
source /opt/robot/current/bin/ros2env
source install/setup.bash
ros2 launch orbbec_camera multi_camera.launch.py
EOF

read -r -d '' CAMERA_VIEW_CMD <<'EOF' || true
source /opt/robot/current/bin/ros2env
ros2 run rqt_image_view rqt_image_view
EOF

read -r -d '' STOP_OLD_NAV_CMD <<'EOF' || true
source /opt/robot/current/bin/ros2env
source /home/niic/robot_a_ws/install/setup.bash
export ROS_LOG_DIR=/root/.ros/log

NAV_CLEANUP_STATUS='__NAV_CLEANUP_STATUS__'
rm -f -- "${NAV_CLEANUP_STATUS}"

cleanup_succeeded=0
for attempt in 1 2 3; do
  echo
  echo "===== 第 ${attempt}/3 轮：执行 daohang.txt 的旧导航关闭流程 ====="

  # 保持与 daohang.txt 相同的精确进程匹配方式。
  LOCALIZATION_PID=$(pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch localization localization.launch.py$' || true)
  NAVIGATION_PID=$(pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch nav2_bringup sai_real_launch.py$' || true)

  echo "定位进程：${LOCALIZATION_PID:-未找到}"
  echo "导航进程：${NAVIGATION_PID:-未找到}"

  [[ -n "${LOCALIZATION_PID}" ]] && kill -INT ${LOCALIZATION_PID} 2>/dev/null || true
  [[ -n "${NAVIGATION_PID}" ]] && kill -INT ${NAVIGATION_PID} 2>/dev/null || true

  echo "等待 8 秒，让节点完成退出……"
  sleep 8

  LOCALIZATION_LEFT=$(pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch localization localization.launch.py$' || true)
  NAVIGATION_LEFT=$(pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch nav2_bringup sai_real_launch.py$' || true)
  CONFLICT_NODES=$(ros2 node list 2>/dev/null | grep -E \
    'fast_lio|ekf_filter_node|localization_node|map_server|controller_server|planner_server' \
    || true)

  echo "===== 第 ${attempt} 轮检查结果 ====="
  echo "剩余定位进程：${LOCALIZATION_LEFT:-无}"
  echo "剩余导航进程：${NAVIGATION_LEFT:-无}"
  if [[ -n "${CONFLICT_NODES}" ]]; then
    echo "剩余冲突节点："
    printf '%s\n' "${CONFLICT_NODES}"
  else
    echo "剩余冲突节点：无"
  fi

  if [[ -z "${LOCALIZATION_LEFT}" && -z "${NAVIGATION_LEFT}" && -z "${CONFLICT_NODES}" ]]; then
    cleanup_succeeded=1
    break
  fi

  if (( attempt < 3 )); then
    echo "旧导航尚未完全关闭，准备再次清理……"
  fi
done

if (( cleanup_succeeded == 1 )); then
  # 人工操作中，从看到“没有冲突节点”到输入下一条命令通常存在短暂间隔。
  # 给 DDS/ROS 图 2 秒完成节点注销，避免雷达 RViz 读到正在变化的图状态。
  echo "旧节点已清空，等待 ROS 图稳定 2 秒……"
  sleep 2
  printf '%s\n' success > "${NAV_CLEANUP_STATUS}"
  echo "===== 旧定位和 Nav2 已完全停止，可以执行 daohang.txt 的雷达命令 ====="
else
  printf '%s\n' failed > "${NAV_CLEANUP_STATUS}"
  echo "错误：连续清理 3 次后仍有旧进程或冲突节点。"
  echo "为避免节点冲突，本次不会启动雷达，请人工检查上面的残留信息。"
  exit 1
fi
EOF

read -r -d '' LIDAR_CMD <<'EOF' || true
NAV_CLEANUP_STATUS='__NAV_CLEANUP_STATUS__'
LIDAR_READY_STATUS='__LIDAR_READY_STATUS__'

echo "等待旧定位和 Nav2 完全关闭……"
for ((wait_count = 0; wait_count < 120; wait_count++)); do
  [[ -f "${NAV_CLEANUP_STATUS}" ]] && break
  sleep 1
done

if [[ ! -f "${NAV_CLEANUP_STATUS}" ]]; then
  echo "错误：等待旧导航关闭超时（120 秒），本次不启动雷达。"
  exit 1
fi

cleanup_result=$(<"${NAV_CLEANUP_STATUS}")
if [[ "${cleanup_result}" != success ]]; then
  echo "错误：旧导航未能完全关闭，本次不启动雷达。"
  exit 1
fi

echo "旧导航已经完全关闭，现在启动雷达节点。"
# 与后续 TF、/scan 标签页使用同一套 ROS 环境，避免 ROS_DOMAIN_ID/RMW 不一致。
source /opt/robot/current/bin/ros2env
source /home/niic/mojl_robotA/install/setup.bash
export ROS_LOG_DIR=/root/.ros/log
echo "雷达 ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}，RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-默认}"

ros2 launch livox_ros_driver2 rviz_MID360s_launch.py &
LIDAR_LAUNCH_PID=$!

stop_lidar_launch() {
  kill -INT "${LIDAR_LAUNCH_PID}" 2>/dev/null || true
}
trap stop_lidar_launch INT TERM

echo "等待雷达实际发布 /livox/lidar……"
lidar_ready=0
for ((check_count = 0; check_count < 300; check_count++)); do
  if ! kill -0 "${LIDAR_LAUNCH_PID}" 2>/dev/null; then
    break
  fi
  if ros2 topic list 2>/dev/null | grep -Fxq /livox/lidar; then
    lidar_ready=1
    break
  fi
  sleep 0.1
done

if (( lidar_ready == 0 )); then
  printf '%s\n' failed > "${LIDAR_READY_STATUS}"
  echo "错误：30 秒内没有发现 /livox/lidar，停止后续 TF 和 /scan。"
  stop_lidar_launch
  wait "${LIDAR_LAUNCH_PID}" 2>/dev/null || true
  exit 1
fi

printf '%s\n' success > "${LIDAR_READY_STATUS}"
echo "雷达点云已经发布，允许继续启动 TF 和 /scan。"
wait "${LIDAR_LAUNCH_PID}"
EOF

read -r -d '' WHEEL_TF_CMD <<'EOF' || true
source /opt/robot/current/bin/ros2env
source /home/niic/robot_a_ws/install/setup.bash
export ROS_LOG_DIR=/root/.ros/log
ros2 run topic_tools relay \
  /mecanum_drive_controller/tf_odometry \
  /tf \
  --ros-args -r __node:=wheel_tf_relay
EOF

read -r -d '' STATIC_TF_CMD <<'EOF' || true
source /opt/robot/current/bin/ros2env
source /home/niic/robot_a_ws/install/setup.bash
export ROS_LOG_DIR=/root/.ros/log
ros2 run tf2_ros static_transform_publisher \
  --x 0 \
  --y 0 \
  --z 0 \
  --roll 0 \
  --pitch 0 \
  --yaw 0 \
  --frame-id base_link \
  --child-frame-id base_footprint
EOF

read -r -d '' SCAN_CMD <<'EOF' || true
source /opt/robot/current/bin/ros2env
source /home/niic/robot_a_ws/install/setup.bash
export ROS_LOG_DIR=/root/.ros/log
ros2 run pointcloud_to_laserscan \
  pointcloud_to_laserscan_node --ros-args \
  -r __node:=livox_to_scan \
  -r cloud_in:=/livox/lidar \
  -r scan:=/scan \
  -p min_height:=-0.10 \
  -p max_height:=2.00 \
  -p range_min:=0.50 \
  -p range_max:=8.0 \
  -p angle_min:=-3.14159 \
  -p angle_max:=3.14159 \
  -p angle_increment:=0.00874
EOF

# 普通标签页收到“已经进入 root 并开始执行”的确认后，仅短暂停顿再启动下一页。
# 可改为 0（最快）或 1（机器负载较高时更稳）。
TAB_START_INTERVAL=0.5
TAB_START_TIMEOUT=15

first_tab=1
tab_script_dir=$(mktemp -d /tmp/robot_a_start_tabs.XXXXXX) || exit 1
chmod 700 "${tab_script_dir}"
nav_cleanup_status="${tab_script_dir}/nav_cleanup.status"
lidar_ready_status="${tab_script_dir}/lidar_ready.status"
tab_started_dir="${tab_script_dir}/started"
mkdir -m 700 "${tab_started_dir}"
STOP_OLD_NAV_CMD="${STOP_OLD_NAV_CMD//__NAV_CLEANUP_STATUS__/${nav_cleanup_status}}"
LIDAR_CMD="${LIDAR_CMD//__NAV_CLEANUP_STATUS__/${nav_cleanup_status}}"
LIDAR_CMD="${LIDAR_CMD//__LIDAR_READY_STATUS__/${lidar_ready_status}}"

# Ubuntu 的 sudo 凭据默认可能按终端分别记录，因此新标签页仍会重复询问密码。
# 这里用只对当前用户开放的临时管道，将刚才验证过的密码交给各标签页的
# SUDO_ASKPASS。密码只经过内存和管道，不会写进临时脚本或命令行参数。
askpass_request_fifo="${tab_script_dir}/askpass_requests"
askpass_helper="${tab_script_dir}/askpass_helper.sh"
mkfifo -m 600 "${askpass_request_fifo}"

cat > "${askpass_helper}" <<EOF
#!/usr/bin/env bash
set -u
auth_dir=$(printf '%q' "${tab_script_dir}")
request_fifo=$(printf '%q' "${askpass_request_fifo}")
reply_fifo="\${auth_dir}/reply.\$\$.\${RANDOM}"
mkfifo -m 600 "\${reply_fifo}" || exit 1
printf '%s\n' "\${reply_fifo}" > "\${request_fifo}"
cat "\${reply_fifo}"
rm -f -- "\${reply_fifo}"
EOF
chmod 700 "${askpass_helper}"

# 保持请求管道打开，避免读取端在两个请求之间收到 EOF。
exec 9<> "${askpass_request_fifo}"

serve_root_password() {
  local served=0 reply_fifo
  while (( served < 10 )); do
    IFS= read -r reply_fifo <&9 || continue

    # 只响应当前私有临时目录中由 askpass 创建的命名管道。
    case "${reply_fifo}" in
      "${tab_script_dir}"/reply.*)
        if [[ -p "${reply_fifo}" ]]; then
          printf '%s\n' "${ROOT_PASSWORD}" > "${reply_fifo}"
          ((served += 1))
        fi
        ;;
    esac
  done
}

serve_root_password &
password_server_pid=$!

add_tab() {
  local title="$1"
  local command_body="$2"
  local tab_number=${title%% *}
  local root_payload wrapper tab_script started_file

  tab_script="${tab_script_dir}/tab_${title// /_}.sh"
  started_file="${tab_started_dir}/${tab_number}.started"
  root_payload='rm -f -- "$0"'$'\n'
  root_payload+="printf '%s\\n' started > $(printf '%q' "${started_file}")"$'\n'
  root_payload+="${command_body}"$'\n\n''tab_status=$?'$'\n'
  root_payload+='echo'$'\n'
  root_payload+="echo \"===== ${title} 已结束，退出状态：\${tab_status} =====\""$'\n'
  root_payload+='echo "当前保留 root 命令行；输入 exit 可关闭此标签页。"'$'\n'
  root_payload+='exec bash -i'

  printf '%s\n' "${root_payload}" > "${tab_script}"
  chmod 700 "${tab_script}"
  # GNOME Terminal 的 --command 不通过 shell 解释，环境变量赋值不能放在命令首位。
  # 使用 /usr/bin/env 作为实际程序，避免把 SUDO_ASKPASS=... 误当成可执行文件。
  # bash -ic 会像手动 sudo su 后的交互式 root shell 一样加载 /root/.bashrc，
  # 再执行标签页脚本；这样各标签页的基础 ROS 环境保持一致。
  printf -v root_command 'exec bash %q' "${tab_script}"
  printf -v wrapper '/usr/bin/env SUDO_ASKPASS=%q sudo -A -H -p "" bash -ic %q' \
    "${askpass_helper}" "${root_command}"

  if [[ ${first_tab} -eq 1 ]]; then
    gnome-terminal --window --maximize --title="${title}" --command="${wrapper}"
    first_tab=0
  else
    gnome-terminal --tab --title="${title}" --command="${wrapper}"
  fi

  # 等到这个标签页已经通过 sudo 进入 root，并真正开始执行，再创建下一页。
  for ((wait_step = 0; wait_step < TAB_START_TIMEOUT * 10; wait_step++)); do
    [[ -f "${started_file}" ]] && break
    sleep 0.1
  done

  if [[ ! -f "${started_file}" ]]; then
    echo "错误：${title} 未能在 ${TAB_START_TIMEOUT} 秒内启动，停止后续标签页。"
    return 1
  fi

  echo "已启动：${title}"
  sleep "${TAB_START_INTERVAL}"
}

# 严格按编号创建标签页。这里只等待“命令已经开始”，不等待长期 ROS 节点退出。
add_tab "01 机械臂驱动" "${ARM_CMD}" || launch_failed=1
[[ ${launch_failed:-0} -eq 0 ]] && add_tab "02 底盘驱动" "${CHASSIS_CMD}" || launch_failed=1
[[ ${launch_failed:-0} -eq 0 ]] && add_tab "03 MoveIt" "${MOVEIT_CMD}" || launch_failed=1
[[ ${launch_failed:-0} -eq 0 ]] && add_tab "04 相机节点" "${CAMERA_CMD}" || launch_failed=1
[[ ${launch_failed:-0} -eq 0 ]] && add_tab "05 相机可视化" "${CAMERA_VIEW_CMD}" || launch_failed=1
[[ ${launch_failed:-0} -eq 0 ]] && add_tab "06 关闭旧导航" "${STOP_OLD_NAV_CMD}" || launch_failed=1

# 第 07 页是特殊依赖：不仅要等第 06 页开始，还必须等它最多三轮清理并明确成功。
if [[ ${launch_failed:-0} -eq 0 ]]; then
  echo "等待旧定位和 Nav2 完全关闭后再创建雷达标签页……"
  for ((wait_step = 0; wait_step < 1200; wait_step++)); do
    [[ -f "${nav_cleanup_status}" ]] && break
    sleep 0.1
  done

  if [[ ! -f "${nav_cleanup_status}" ]]; then
    echo "错误：等待旧导航清理结果超时，停止启动第 07～10 页。"
    launch_failed=1
  elif [[ $(<"${nav_cleanup_status}") != success ]]; then
    echo "错误：旧导航未完全关闭，停止启动第 07～10 页。"
    launch_failed=1
  fi
fi

[[ ${launch_failed:-0} -eq 0 ]] && add_tab "07 雷达节点" "${LIDAR_CMD}" || launch_failed=1

# 雷达标签页的命令已经开始并不代表驱动已就绪；必须看到 /livox/lidar 才启动 TF。
if [[ ${launch_failed:-0} -eq 0 ]]; then
  echo "等待雷达驱动确认 /livox/lidar 已发布……"
  for ((wait_step = 0; wait_step < 300; wait_step++)); do
    [[ -f "${lidar_ready_status}" ]] && break
    sleep 0.1
  done

  if [[ ! -f "${lidar_ready_status}" ]]; then
    echo "错误：等待雷达就绪超时，停止启动第 08～10 页。"
    launch_failed=1
  elif [[ $(<"${lidar_ready_status}") != success ]]; then
    echo "错误：雷达没有成功发布点云，停止启动第 08～10 页。"
    launch_failed=1
  fi
fi

[[ ${launch_failed:-0} -eq 0 ]] && add_tab "08 轮式里程计 TF" "${WHEEL_TF_CMD}" || launch_failed=1
[[ ${launch_failed:-0} -eq 0 ]] && add_tab "09 静态 TF" "${STATIC_TF_CMD}" || launch_failed=1
[[ ${launch_failed:-0} -eq 0 ]] && add_tab "10 发布 scan" "${SCAN_CMD}" || launch_failed=1

# 最多等待 30 秒，让 10 个标签页完成自动 sudo 验证。
if [[ ${launch_failed:-0} -ne 0 ]]; then
  kill "${password_server_pid}" 2>/dev/null || true
else
  for ((i = 0; i < 300; i++)); do
    kill -0 "${password_server_pid}" 2>/dev/null || break
    sleep 0.1
  done
fi

if kill -0 "${password_server_pid}" 2>/dev/null; then
  echo "警告：部分标签页没有在 30 秒内完成 root 验证。"
  kill "${password_server_pid}" 2>/dev/null || true
fi
wait "${password_server_pid}" 2>/dev/null || true

unset ROOT_PASSWORD
exec 9>&-
exec 9<&-
rm -f -- "${askpass_helper}" "${askpass_request_fifo}"
rm -f -- "${tab_started_dir}"/*.started "${nav_cleanup_status}" "${lidar_ready_status}"
rmdir "${tab_started_dir}" 2>/dev/null || true
rmdir "${tab_script_dir}" 2>/dev/null || true

if [[ ${launch_failed:-0} -ne 0 ]]; then
  echo "启动链未完成，请根据最后一个错误标签页检查问题。"
  exit 1
fi
