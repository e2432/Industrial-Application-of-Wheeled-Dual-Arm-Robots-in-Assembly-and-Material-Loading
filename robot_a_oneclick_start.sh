#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/niic/project/20260623/20260116/test_robot_A"
ROS_ENV="/opt/robot/current/bin/ros2env"

CMD1="cd '$PROJECT' && source '$ROS_ENV' && source install/setup.bash && exec ros2 run upper_ethercat ethercat-upper -x 6 -l 0"
CMD2="cd '$PROJECT' && source '$ROS_ENV' && source install/setup.bash && exec ros2 launch ethercat_chassis ethercat_chassis.launch.py"
CMD3="cd '$PROJECT' && source '$ROS_ENV' && source install/setup.bash && exec ros2 launch robot_a_moveit_config demo.launch.py"

echo "============================================================"
echo " Robot A 一键启动"
echo "============================================================"
echo "需要 sudo 权限，只验证一次密码。"
echo

sudo -v

(
    while true; do
        sudo -n true 2>/dev/null || exit
        sleep 30
    done
) &
SUDO_KEEPALIVE_PID=$!

cleanup_keepalive() {
    kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
}
trap cleanup_keepalive EXIT

start_gnome_terminal() {
    echo "[1/3] 启动 upper_ethercat"
    gnome-terminal --title="Robot A - upper_ethercat" --         bash -lc "sudo bash -lc \"$CMD1\"; rc=\$?; echo; echo '[upper_ethercat] exited, code='\$rc; exec bash"

    sleep 3

    echo "[2/3] 启动 ethercat_chassis"
    gnome-terminal --title="Robot A - ethercat_chassis" --         bash -lc "sudo bash -lc \"$CMD2\"; rc=\$?; echo; echo '[ethercat_chassis] exited, code='\$rc; exec bash"

    sleep 3

    echo "[3/3] 启动 MoveIt"
    gnome-terminal --title="Robot A - MoveIt" --         bash -lc "sudo bash -lc \"$CMD3\"; rc=\$?; echo; echo '[MoveIt] exited, code='\$rc; exec bash"
}

start_xterm() {
    echo "[1/3] 启动 upper_ethercat"
    xterm -T "Robot A - upper_ethercat"         -e bash -lc "sudo bash -lc \"$CMD1\"; rc=\$?; echo; echo '[upper_ethercat] exited, code='\$rc; exec bash" &

    sleep 3

    echo "[2/3] 启动 ethercat_chassis"
    xterm -T "Robot A - ethercat_chassis"         -e bash -lc "sudo bash -lc \"$CMD2\"; rc=\$?; echo; echo '[ethercat_chassis] exited, code='\$rc; exec bash" &

    sleep 3

    echo "[3/3] 启动 MoveIt"
    xterm -T "Robot A - MoveIt"         -e bash -lc "sudo bash -lc \"$CMD3\"; rc=\$?; echo; echo '[MoveIt] exited, code='\$rc; exec bash" &
}

start_background() {
    LOG_DIR="/tmp/robot_a_start_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"

    echo "[WARN] 未找到图形终端，改为后台启动"
    echo "[INFO] 日志目录: $LOG_DIR"

    sudo bash -lc "$CMD1" >"$LOG_DIR/upper_ethercat.log" 2>&1 &
    PID1=$!
    echo "$PID1" >"$LOG_DIR/upper_ethercat.pid"

    sleep 3

    sudo bash -lc "$CMD2" >"$LOG_DIR/ethercat_chassis.log" 2>&1 &
    PID2=$!
    echo "$PID2" >"$LOG_DIR/ethercat_chassis.pid"

    sleep 3

    sudo bash -lc "$CMD3" >"$LOG_DIR/moveit.log" 2>&1 &
    PID3=$!
    echo "$PID3" >"$LOG_DIR/moveit.pid"

    echo "后台进程已启动:"
    echo "  upper_ethercat PID=$PID1"
    echo "  chassis        PID=$PID2"
    echo "  MoveIt         PID=$PID3"
    echo
    echo "查看日志:"
    echo "  tail -f $LOG_DIR/upper_ethercat.log"
    echo "  tail -f $LOG_DIR/ethercat_chassis.log"
    echo "  tail -f $LOG_DIR/moveit.log"
}

if command -v gnome-terminal >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
    start_gnome_terminal
elif command -v xterm >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
    start_xterm
else
    start_background
fi

echo
echo "============================================================"
echo " Robot A 启动命令已全部下发"
echo "============================================================"
