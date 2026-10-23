#!/usr/bin/env bash

BASE=/home/niic/Desktop/competition/wheel/livox_ws

source /opt/robot/current/bin/ros2env
source "$BASE/install/setup.bash"

export LD_LIBRARY_PATH="$BASE/sdk2_install/lib:${LD_LIBRARY_PATH:-}"
export ROS_HOME="$BASE/ros_home"
export ROS_LOG_DIR="$BASE/log/ros"

mkdir -p "$ROS_HOME" "$ROS_LOG_DIR"
