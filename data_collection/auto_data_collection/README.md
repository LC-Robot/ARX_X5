# 自动数据采集 - 带随机性的轨迹重放

## 功能说明

`data_replay.py` 可以读取关键点文件（waypoint1.txt），并生成带有随机性的平滑轨迹来控制机械臂。

### 主要特点

1. **三次样条插值**：在关键点之间生成平滑轨迹
2. **随机性注入**：每次运行都会产生略微不同的轨迹
3. **保持关键点**：所有关键点位置保持不变，只在中间添加随机性
4. **夹爪特殊处理**：夹爪的随机性较小，确保抓取成功

## 使用方法

### 基本用法

```bash
python data_replay.py X5 can0 waypoint1.txt
```

### 自定义参数

```bash
# 调整随机性程度（0.0-1.0，默认0.05）
python data_replay.py X5 can0 waypoint1.txt --randomness 0.1

# 调整插值点数量（默认每段10个点）
python data_replay.py X5 can0 waypoint1.txt --points-per-segment 15

# 可视化轨迹（需要matplotlib）
python data_replay.py X5 can0 waypoint1.txt --visualize

# 重复执行多次（用于数据采集）
python data_replay.py X5 can0 waypoint1.txt --repeat 5

# 组合使用
python data_replay.py X5 can0 waypoint1.txt --randomness 0.08 --points-per-segment 12 --repeat 3 --visualize
```

## 参数说明

### 必需参数
- `model`: 机械臂型号（X5, L5, X7等）
- `interface`: CAN接口名称（can0, can1等）
- `waypoint_file`: 关键点文件路径

### 可选参数
- `--randomness`: 随机性程度（默认：0.05）
  - `0.0`: 无随机性，纯插值
  - `0.05`: 低随机性（默认，推荐用于抓取任务）
  - `0.1`: 中等随机性
  - `0.2+`: 高随机性（可能影响任务成功率）
  
- `--points-per-segment`: 每两个关键点之间的插值点数量（默认：10）
  - 越大越平滑，但执行时间越长
  - 推荐范围：5-20
  
- `--visualize`: 执行前可视化轨迹（需要安装matplotlib）
  - 显示关键点和插值后的完整轨迹
  - 可以看到随机性的效果
  
- `--repeat`: 重复执行次数（默认：1）
  - 用于批量数据采集
  - 每次执行都会生成不同的随机轨迹

## 文件格式

`waypoint1.txt` 格式：
```
joint0,joint1,joint2,joint3,joint4,joint5,gripper
-0.282483,0.931373,1.079385,-1.068322,-0.186732,-0.224117,0.000390
-0.237850,0.946631,0.778400,-0.755131,-0.153543,-0.185969,0.000390
...
```

## 随机性原理

1. **平滑噪声**：使用多个低频正弦波叠加，产生平滑的随机扰动
2. **幅度控制**：根据每个关节的运动范围动态调整噪声幅度
3. **关键点固定**：确保所有关键点保持原始位置不变
4. **夹爪保护**：夹爪的随机性设置为其他关节的30%

## 注意事项

1. 首次运行时建议使用较小的随机性（0.03-0.05）
2. 确保关键点文件路径正确
3. 运行前确保机械臂周围无障碍物
4. 每次运行都会产生不同的轨迹

