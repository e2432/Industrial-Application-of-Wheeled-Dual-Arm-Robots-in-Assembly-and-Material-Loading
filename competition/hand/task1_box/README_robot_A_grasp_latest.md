
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


