# Robot A 双臂搬箱抓取控制 README

> 适用脚本：`dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py`  
> 运行环境：ROS 2 Humble、Robot A 真实机器人、`ros2_control`  
> 推荐目录：`/home/niic/Desktop/competition/hand/`  
> 文档版本：2026-08-06

---

## 1. 功能概述

该脚本用于控制 Robot A 的双机械臂从箱体两侧进行夹紧抓取，并在抓取完成后通过升降机构 `lift` 将箱体抬升到指定的地面参考高度。

最新版具有以下功能：

- 读取真实机器人 `/joint_states`；
- 连接 8 个真实机器人 Action 控制器；
- 从 URDF 读取关节位置和速度限位；
- 使用本地 FK 和数值雅可比 IK；
- 将 XY 平面定位与 Z 高度定位分开；
- 保持前臂接近水平；
- 保持夹爪接近竖直；
- 使关节姿态接近标准抓取模板；
- 提高箱体左右夹紧方向对输入 X 的响应；
- 使用 `arm_*2_arm_*3_joint` 进行横向夹紧精修；
- 根据 XY 位姿的 FK 结果单独计算抓取 lift；
- 夹爪闭合后，将 lift 升高到指定地面参考高度；
- 默认只进行 IK、FK、限位和控制器检查；
- 只有显式添加 `--execute` 才会让机器人运动。

**脚本不包含完整的自碰撞、环境碰撞、箱体受力和防滑检测。**

---

## 2. 文件位置

将最新版脚本放在：

```text
/home/niic/Desktop/competition/hand/dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py
```

README 建议放在同一目录：

```text
/home/niic/Desktop/competition/hand/README_robot_A_grasp.md
```

检查文件：

```bash
cd /home/niic/Desktop/competition/hand

ls -lh \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py
```

添加执行权限并检查语法：

```bash
chmod +x \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py

/usr/bin/python3.10 -m py_compile \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py
```

语法检查没有输出表示通过。

---

## 3. ROS 环境配置

每次新开终端后执行：

```bash
source /opt/ros/humble/setup.bash
source /opt/robot/current/bin/ros2env
source /home/niic/project/20260623/20260116/test_robot_A/install/setup.bash
```

进入程序目录：

```bash
cd /home/niic/Desktop/competition/hand
```

该脚本推荐使用系统 Python：

```bash
/usr/bin/python3.10
```

即使当前终端显示已进入 `.venv`，运行脚本时仍建议显式使用：

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py
```

---

## 4. 实机控制器要求

程序启动后会等待以下控制器：

```text
/lift_controller/follow_joint_trajectory
/waist_controller/follow_joint_trajectory
/left_arm_root_controller/follow_joint_trajectory
/right_arm_root_controller/follow_joint_trajectory
/left_arm_controller/follow_joint_trajectory
/right_arm_controller/follow_joint_trajectory
/left_gripper_controller/gripper_cmd
/right_gripper_controller/gripper_cmd
```

对应 Action 类型：

```text
FollowJointTrajectory
GripperCommand
```

可在运行前检查：

```bash
ros2 action list | grep -E \
'lift_controller|waist_controller|arm_root_controller|arm_controller|gripper_controller'
```

程序还需要 `/joint_states` 提供全部 16 个运动关节的状态：

```bash
ros2 topic echo /joint_states --once
```

程序默认最多等待控制器和完整关节状态约 8 秒。

---

## 5. 关节分组

### 5.1 身体关节

```text
base_elevator_joint
elevator_pitch_joint
```

### 5.2 左臂

```text
pitch_arm_left1_joint
arm_left1_arm_left2_joint
arm_left2_arm_left3_joint
arm_left3_arm_left4_joint
arm_left4_arm_left5_joint
arm_left5_arm_left6_joint
arm_left6_arm_left7_joint
```

### 5.3 右臂

```text
pitch_arm_right1_joint
arm_right1_arm_right2_joint
arm_right2_arm_right3_joint
arm_right3_arm_right4_joint
arm_right4_arm_right5_joint
arm_right5_arm_right6_joint
arm_right6_arm_right7_joint
```

横向夹紧精修主要使用：

```text
左臂：arm_left2_arm_left3_joint
右臂：arm_right2_arm_right3_joint
```

右臂向箱体内侧夹紧时，该关节通常会减小；左臂做镜像调整。程序通过 FK 搜索自动判断方向，不直接硬编码正负方向。

---

## 6. 坐标约定

命令行输入使用 Isaac 坐标：

```text
X：左右方向
Y：前后方向
Z：相对地面的高度
```

默认约定：

```text
左侧 X 为负
右侧 X 为正
```

例如：

```text
左夹爪目标：(-0.30, 0.65, 0.95)
右夹爪目标：( 0.30, 0.65, 0.95)
```

脚本内部转换为 ROS/FK 坐标：

```text
ROS X = Isaac Y
ROS Y = -Isaac X
ROS Z = Isaac Z
```

代码中的转换关系是：

```python
return (y, -x, z)
```

因此：

- Isaac X 是箱体左右夹紧方向；
- Isaac Y 是机械臂前后伸出方向；
- Isaac Z 由 lift 独立定位。

---

## 7. 标准抓取模板

默认 7 关节模板顺序：

```text
pitch_arm_*1_joint
arm_*1_arm_*2_joint
arm_*2_arm_*3_joint
arm_*3_arm_*4_joint
arm_*4_arm_*5_joint
arm_*5_arm_*6_joint
arm_*6_arm_*7_joint
```

左臂默认模板：

```text
[-20°, 0°, 0°, -70°, -90°, 0°, 0°]
```

右臂默认模板：

```text
[20°, 0°, 0°, 70°, 90°, 0°, 0°]
```

模板用于约束：

- 前臂接近水平；
- 夹爪接近竖直；
- 双臂姿态接近标准搬箱姿态；
- 避免 IK 只满足位置而产生不适合夹箱的关节构型。

模板是优化参考，不要求每个关节与模板完全相同。

---

## 8. 核心定位逻辑

### 8.1 XY 平面 IK

程序首先固定当前 lift，只求解机械臂在 XY 平面的位置。

主 IK 同时优化：

- XY 平面误差；
- 标准模板末端方向；
- 前臂水平程度；
- 夹爪竖直程度；
- 关节与标准模板的接近程度。

目标 Z 不参与机械臂关节 IK。

### 8.2 箱体夹紧方向精修

Isaac X 对应内部 ROS Y，是左右夹紧箱体的方向。

程序通过两层机制提高 X 方向敏感性：

1. 使用 `--lateral-weight` 提高横向误差在主 IK 中的权重；
2. 主 IK 后，单独扫描和精修：
   - `arm_left2_arm_left3_joint`
   - `arm_right2_arm_right3_joint`

精修过程中仍会检查：

- 前后方向误差；
- 前臂水平；
- 夹爪竖直；
- 模板方向；
- 模板关节偏差；
- 真实 URDF 关节限位。

### 8.3 Z 与 lift 独立定位

XY IK 完成后，程序使用 FK 得到当前机械臂姿态下的左右夹爪 Z：

```text
left_gripper_z
right_gripper_z
```

分别计算左右臂所需抓取 lift：

```text
required_lift
= 当前 lift + 目标 Z - XY 解对应的夹爪 Z
```

Robot A 左右臂共用一个 lift，因此程序会检查左右需求是否一致，并取平均值作为抓取 lift。

### 8.4 抓取后的指定高度

提供：

```bash
--final-ground-height HEIGHT
```

后，程序会根据抓取完成时的夹爪实际 FK 高度计算最终 lift：

```text
最终 lift
= 抓取 lift
+ 指定地面高度
- 抓取时夹爪实际高度
```

`--final-ground-height` 表示：

```text
夹爪中心相对地面的最终高度，单位为米
```

它不是 lift 关节值。

例如：

```bash
--final-ground-height 1.10
```

表示抓取完成并闭合夹爪后，将夹爪中心升高到距离地面约 1.10 米。

程序只允许该阶段“升高”。如果指定高度低于抓取时夹爪高度，程序会拒绝执行。

---

## 9. 完整动作流程

提供 `--final-ground-height` 时，动作共 7 个阶段：

```text
[1/7] 夹爪张开

[2/7] 肩部避障抬举
      左肩 = 最终左肩角 + shoulder_clearance
      右肩 = 最终右肩角 - shoulder_clearance

[3/7] 其他机械臂关节运动到 XY 抓取位姿
      肩部保持避障角度

[4/7] lift 调整到抓取 Z

[5/7] 肩部逐渐下降到最终抓取角

[6/7] 夹爪闭合

[7/7] lift 升高
      使夹爪中心达到 final-ground-height
```

不提供 `--final-ground-height` 时，只执行前 6 个阶段。

不提供 `--close-gripper` 时，第 6 阶段会保持夹爪张开。即使指定了最终高度，程序仍可能在夹爪张开的情况下执行第 7 阶段。因此，正式搬箱命令必须同时提供：

```bash
--execute
--close-gripper
--final-ground-height ...
```

---

## 10. 推荐操作流程

### 10.1 第一步：只进行计算和安全检查

下面的命令不会让机器人运动：

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l -0.30 0.65 0.95 \
  -r 0.30 0.65 0.95 \
  --velocity-scale 0.50 \
  --lateral-weight 4.0 \
  --final-ground-height 1.10
```

必须确认输出包含：

```text
XY/Z 独立定位与安全检查通过
机器人没有运动
```

并检查：

- 左右控制器全部就绪；
- 16 个关节状态已同步；
- XY 误差合格；
- 夹紧 X 误差合格；
- Z 误差合格；
- 前臂倾角合格；
- 夹爪竖直误差合格；
- 抓取 lift 未越界；
- 最终升高 lift 未越界；
- `lift_delta_m` 为非负；
- 最终夹爪高度接近指定高度。

### 10.2 第二步：实机路径测试，不闭合夹爪

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l -0.30 0.65 0.95 \
  -r 0.30 0.65 0.95 \
  --velocity-scale 0.30 \
  --lateral-weight 4.0 \
  --final-ground-height 1.10 \
  --post-lift-duration 4.0 \
  --execute
```

注意：该命令会让机器人运动，并在夹爪保持张开的情况下执行最后升高，仅用于无负载路径检查。

重点观察：

- 肩部抬举方向；
- 双臂是否撞到机身；
- 双臂是否互相碰撞；
- 夹爪是否对准箱体两侧；
- lift 运动方向；
- 抓取后升高路径；
- 线缆是否受拉扯。

### 10.3 第三步：正式夹箱、闭合并升高

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l -0.30 0.65 0.95 \
  -r 0.30 0.65 0.95 \
  --velocity-scale 0.50 \
  --lateral-weight 4.0 \
  --final-ground-height 1.10 \
  --post-lift-duration 3.0 \
  --execute \
  --close-gripper
```

这是完整搬箱流程。

---

## 11. 常用命令示例

### 11.1 只进行 IK 和安全检查

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l -0.30 0.65 0.95 \
  -r 0.30 0.65 0.95
```

### 11.2 检查抓取并规划最终高度

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l -0.30 0.65 0.95 \
  -r 0.30 0.65 0.95 \
  --final-ground-height 1.10
```

### 11.3 更慢的首次实机测试

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l -0.30 0.65 0.95 \
  -r 0.30 0.65 0.95 \
  --velocity-scale 0.25 \
  --final-ground-height 1.10 \
  --post-lift-duration 4.0 \
  --execute
```

### 11.4 正式闭合夹爪但不执行最终升高

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l -0.30 0.65 0.95 \
  -r 0.30 0.65 0.95 \
  --velocity-scale 0.50 \
  --execute \
  --close-gripper
```

### 11.5 正式抓取并升高到 1.10 米

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l -0.30 0.65 0.95 \
  -r 0.30 0.65 0.95 \
  --velocity-scale 0.50 \
  --lateral-weight 4.0 \
  --final-ground-height 1.10 \
  --execute \
  --close-gripper
```

### 11.6 改变箱体宽度方向目标

更靠近机器人中心：

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l -0.27 0.65 0.95 \
  -r 0.27 0.65 0.95 \
  --final-ground-height 1.10
```

更向两侧展开：

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l -0.33 0.65 0.95 \
  -r 0.33 0.65 0.95 \
  --final-ground-height 1.10
```

每次修改目标后都必须先执行不带 `--execute` 的安全检查。

---

## 12. 参数说明

| 参数 | 默认值 | 单位 | 说明 |
|---|---:|---|---|
| `-l / --left X Y Z` | `-0.30 0.65 0.95` | m | 左夹爪 Isaac 坐标目标 |
| `-r / --right X Y Z` | `0.30 0.65 0.95` | m | 右夹爪 Isaac 坐标目标 |
| `--left-template-deg` | `-20 0 0 -70 -90 0 0` | deg | 左臂标准模板 |
| `--right-template-deg` | `20 0 0 70 90 0 0` | deg | 右臂标准模板 |
| `--urdf` | Robot A 默认 URDF 路径 | — | 读取真实关节位置和速度限位 |
| `--velocity-scale` | `0.50` | 比例 | URDF 最大速度缩放，范围 `(0,1]` |
| `--max-xy-error-cm` | `1.0` | cm | 最终 XY 平面最大误差 |
| `--max-z-error-cm` | `1.0` | cm | 抓取 lift 调整后的最大 Z 误差 |
| `--lateral-weight` | `4.0` | 权重 | Isaac X/夹紧方向在主 IK 中的权重 |
| `--max-lateral-error-cm` | `0.5` | cm | q2 精修后的夹紧方向最大误差 |
| `--max-lateral-joint-correction-deg` | `20.0` | deg | q2 相对主 IK 解的最大精修角度 |
| `--max-shared-lift-mismatch-cm` | `1.0` | cm | 左右臂所需共享 lift 的最大差值 |
| `--max-orientation-error-deg` | `5.0` | deg | 标准模板末端方向最大误差 |
| `--max-forearm-tilt-deg` | `3.0` | deg | 前臂最大倾角 |
| `--max-gripper-vertical-error-deg` | `5.0` | deg | 夹爪竖直方向最大误差 |
| `--max-template-rmse-deg` | `20.0` | deg | 关节相对模板的最大 RMSE |
| `--shoulder-clearance` | `0.20` | rad | 肩部相对最终角的避障抬举量 |
| `--final-ground-height` | 未启用 | m | 抓取后夹爪中心相对地面的高度 |
| `--max-final-height-error-cm` | `1.0` | cm | 最终升高后的高度误差上限 |
| `--post-lift-duration` | `2.5` | s | 最后 lift 升高动作时间 |
| `--gripper-effort` | `50.0` | 控制器单位 | 夹爪最大执行力参数 |
| `--execute` | 关闭 | — | 允许真实机器人运动 |
| `--close-gripper` | 关闭 | — | 第 6 阶段闭合夹爪 |

查看程序内置帮助：

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  --help
```

---

## 13. 参数调节建议

### 13.1 X 方向响应仍然不足

先提高：

```bash
--lateral-weight 6.0
```

仍不足时，可适当增加 q2 精修范围：

```bash
--max-lateral-joint-correction-deg 25
```

每次修改后必须先做 dry-run。

不建议一次大幅放宽到很高角度，以免破坏标准模板和双臂安全间距。

### 13.2 目标位置较小，需要更高精度

可以收紧：

```bash
--max-lateral-error-cm 0.3
--max-xy-error-cm 0.7
```

阈值越严格，IK 越可能无解。

### 13.3 首次实机运行

建议：

```bash
--velocity-scale 0.25
--post-lift-duration 4.0
```

确认动作方向和路径后再逐步提高速度。

### 13.4 肩部避障量

默认：

```bash
--shoulder-clearance 0.20
```

该参数是相对最终肩部角度的附加抬举量，不是绝对肩部角度。

---

## 14. 输出结果说明

成功的 dry-run 通常包含：

```text
✅ 升降控制器就绪
✅ 腰部控制器就绪
✅ 左臂根部控制器就绪
✅ 右臂根部控制器就绪
✅ 左臂控制器就绪
✅ 右臂控制器就绪
✅ 左夹爪控制器就绪
✅ 右夹爪控制器就绪
✅ 已同步 16 个运动关节
```

IK 输出重点：

```text
XY 解时夹爪 Z
由 Z 独立计算 lift
arm_left2_arm_left3_joint 横向精修
arm_right2_arm_right3_joint 横向精修
```

最终指标：

```text
XY误差
前后误差
夹紧X误差
Z误差
模板方向误差
前臂倾角
夹爪竖直误差
模板RMSE
```

启用最终高度后还会输出：

```text
抓取后最终升高计划
desired_ground_height_m
grasp_lift_m
final_lift_m
lift_delta_m
left_final_z_m
right_final_z_m
```

其中：

- `grasp_lift_m`：抓取阶段 lift；
- `final_lift_m`：抓取后最终 lift；
- `lift_delta_m`：最后升高距离；
- `left_final_z_m/right_final_z_m`：最终左右夹爪地面高度。

---

## 15. 安全机制

脚本执行前会检查：

- URDF 文件是否存在；
- 16 个运动关节是否都有位置、速度限位；
- 8 个 Action 控制器是否在线；
- `/joint_states` 是否包含全部关节；
- `velocity-scale` 是否位于 `(0,1]`；
- IK 结果是否处于真实关节限位内；
- 肩部避障角是否越界；
- XY、横向、Z、姿态和模板误差是否合格；
- 左右臂所需 lift 是否一致；
- 抓取 lift 是否越界；
- 最终升高 lift 是否越界；
- 最终高度是否低于抓取高度；
- 最终夹爪高度误差是否合格；
- 轨迹持续时间是否满足 URDF 速度限制。

轨迹使用 smoothstep 插值，并根据关节位移和 URDF 最大速度自动延长持续时间。

---

## 16. 实机安全要求

实机运行前必须做到：

- 实体急停保持可立即触达；
- 机器人周围无人；
- 箱体、桌面、机身和线缆路径明确；
- 没有其他节点同时向相同控制器发送目标；
- 确认夹爪闭合不会夹到人员或线缆；
- 确认 lift 升高后箱体不会碰到上方障碍物；
- 确认箱体重量不超过机器人和夹爪承载能力；
- 先 dry-run，再无负载路径测试，再正式抓取；
- 第一次正式抓取使用较低 `velocity-scale`。

`Ctrl+C` 会请求取消活动轨迹，但不等同于硬件急停。存在碰撞、失控或夹持风险时，应使用实体急停。

---

## 17. 常见问题

### 17.1 找不到脚本

报错：

```text
No such file or directory
```

检查：

```bash
cd /home/niic/Desktop/competition/hand

ls -lh \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py
```

### 17.2 控制器未就绪

报错：

```text
部分实机控制器未就绪
```

检查：

```bash
ros2 action list
ros2 control list_controllers
```

确保机器人驱动和 `ros2_control` 已启动。

### 17.3 `/joint_states` 缺少关节

检查：

```bash
ros2 topic echo /joint_states --once
```

脚本必须收到全部 16 个运动关节。

### 17.4 waist 零点出现微小负值

编码器在零点附近可能反馈：

```text
-0.001 rad
```

最新版会将该值裁剪到 URDF 合法范围，并输出类似：

```text
waist 当前反馈已按 URDF 限位修正
```

这是零点附近的小幅修正，不是放宽 URDF 限位。

### 17.5 XY 模板 IK 无解

可能原因：

- 目标超出工作空间；
- 目标与标准模板冲突；
- 前臂水平或夹爪竖直阈值过严；
- 箱体横向位置要求过严；
- lift 或 waist 可行范围不足。

优先调整目标位置，不要首先大幅放宽安全阈值。

### 17.6 横向夹紧精修失败

可能输出：

```text
左臂横向夹紧关节精修失败
右臂横向夹紧关节精修失败
```

可尝试：

```bash
--lateral-weight 6.0
--max-lateral-joint-correction-deg 25
```

也可以适当调整左右目标 X。

### 17.7 左右 lift 需求不一致

报错表示左右机械臂当前姿态下无法通过同一个 lift 同时达到各自目标 Z。

检查：

- 左右目标 Z 是否一致；
- 左右模板是否对称；
- 左右 X、Y 是否合理对称；
- FK 参数和实际机器人是否一致。

### 17.8 最终高度对应的 lift 越界

说明指定地面高度超出了 lift 的真实行程。

降低：

```bash
--final-ground-height
```

不要修改或绕过 URDF 限位。

### 17.9 指定最终高度低于抓取高度

程序要求最后一步是“升高”，因此不会接受更低的最终高度。

提高 `--final-ground-height`，或省略该参数。

### 17.10 夹爪没有闭合

正式搬箱必须添加：

```bash
--close-gripper
```

并且必须同时有：

```bash
--execute
```

### 17.11 最后没有升高

必须提供：

```bash
--final-ground-height HEIGHT
```

只添加 `--execute --close-gripper` 不会启用第 7 阶段。

---

## 18. 推荐正式命令模板

将下面的目标位置和最终高度替换为实际测量值：

```bash
source /opt/ros/humble/setup.bash
source /opt/robot/current/bin/ros2env
source /home/niic/project/20260623/20260116/test_robot_A/install/setup.bash

cd /home/niic/Desktop/competition/hand

/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l LEFT_X LEFT_Y LEFT_Z \
  -r RIGHT_X RIGHT_Y RIGHT_Z \
  --velocity-scale 0.50 \
  --lateral-weight 4.0 \
  --max-lateral-error-cm 0.5 \
  --max-lateral-joint-correction-deg 20 \
  --shoulder-clearance 0.20 \
  --final-ground-height FINAL_HEIGHT \
  --post-lift-duration 3.0 \
  --execute \
  --close-gripper
```

示例：

```bash
/usr/bin/python3.10 \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  -l -0.30 0.65 0.95 \
  -r 0.30 0.65 0.95 \
  --velocity-scale 0.50 \
  --lateral-weight 4.0 \
  --max-lateral-error-cm 0.5 \
  --max-lateral-joint-correction-deg 20 \
  --shoulder-clearance 0.20 \
  --final-ground-height 1.10 \
  --post-lift-duration 3.0 \
  --execute \
  --close-gripper
```

---

## 19. 运行前最终检查表

正式执行前逐项确认：

- [ ] ROS 2 Humble 环境已加载；
- [ ] Robot A 工作空间已加载；
- [ ] URDF 路径正确；
- [ ] 8 个 Action 控制器全部在线；
- [ ] `/joint_states` 包含全部 16 个关节；
- [ ] 目标左右 X 符号正确；
- [ ] 目标 Y、Z 已实测；
- [ ] `final-ground-height` 是夹爪中心地面高度，不是 lift 值；
- [ ] dry-run 全部通过；
- [ ] 无负载路径测试通过；
- [ ] 夹爪闭合方向正确；
- [ ] lift 上升方向正确；
- [ ] 上方无障碍物；
- [ ] 箱体重量和尺寸符合机器人能力；
- [ ] 没有其他程序控制同一组关节；
- [ ] 实体急停可立即触达；
- [ ] 正式命令包含 `--execute`；
- [ ] 正式搬箱命令包含 `--close-gripper`；
- [ ] 最终升高命令包含 `--final-ground-height`。

---

## 20. 备份建议

修改脚本前先备份：

```bash
cd /home/niic/Desktop/competition/hand

cp \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py.backup
```

恢复：

```bash
cp \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py.backup \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py
```

每次修改后重新检查：

```bash
/usr/bin/python3.10 -m py_compile \
  dual_arm_fixed_grasp_xy_lift_squeeze_raise_real.py
```

---

## 21. 注意事项

1. `final-ground-height` 的准确性依赖当前 FK/URDF 中地面零点、基座高度和机械臂几何参数与真实机器人一致。
2. 如果实际夹爪高度与计算高度存在固定偏差，应校准机器人模型或地面参考系，而不是绕过关节限位。
3. 箱体夹紧效果不仅取决于末端位置，还取决于夹爪接触面、箱体材料、载荷、摩擦和夹爪力。
4. 程序中的姿态检查不能代替实际碰撞检测。
5. 正式运行前必须进行低速、无负载路径验证。
