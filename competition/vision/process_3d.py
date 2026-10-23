"""
depth_test1: YOLO Pose推理 → 短边中点 → 像素→相机系3D坐标
- 输入: color/*.png + depth_aligned_mm/*.png
- 模型: test5 best.pt (YOLO11s, 1280, 4关键点)
- 输出: results_3d.csv (每帧短边中点3D坐标)
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import csv

BASE = Path(__file__).parent
COLOR_DIR = BASE / 'color'
DEPTH_DIR = BASE / 'depth_aligned_mm'
OUTPUT_DIR = BASE / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 相机内参 (来自 Gemini 335Lg 彩色传感器, 1280×720)
# ============================================================
FX = 610.1562
FY = 610.3954
CX = 641.0089
CY = 359.7117
IMG_W, IMG_H = 1280, 720

# ============================================================
# 关键点顺序: 0=rear_left, 1=rear_right, 2=front_left, 3=front_right
# 四条边:
#   rear edge:  0→1  (rear_left → rear_right)
#   front edge: 2→3  (front_left → front_right)
#   left edge:  0→2  (rear_left → front_left)
#   right edge: 1→3  (rear_right → front_right)
# ============================================================

def pixel_to_3d(u, v, depth_mm):
    """像素坐标 + 深度(mm) → 相机坐标系 3D (mm)"""
    z = depth_mm
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return np.array([x, y, z])

def edge_length(p1, p2):
    """两个像素点的欧氏距离"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

KP_NAMES = ['rear_left', 'rear_right', 'front_left', 'front_right']
KP_COLORS = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0)]  # BGR
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3)]  # rear, left, right, front
EDGE_NAMES = ['rear', 'left', 'right', 'front']


def process_frame(color_path, depth_path, model, vis_dir=None):
    """处理单帧: 推理 → 短边中点 → 3D坐标"""
    # 读取图片
    color = cv2.imread(str(color_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        return None

    # 推理
    results = model(color, conf=0.5, imgsz=1280, device=0, verbose=False)
    r = results[0]

    if r.keypoints is None or r.keypoints.data is None or len(r.keypoints.data) == 0:
        return None

    kpts = r.keypoints.data[0]  # 第一个检测结果
    if len(kpts) < 4:
        return None

    # 提取4个角点像素坐标
    pts = {}
    names = ['rear_left', 'rear_right', 'front_left', 'front_right']
    for i, name in enumerate(names):
        kp = kpts[i]
        if kp[2] < 0.5:  # 可见性检查
            return None
        pts[name] = (float(kp[0]), float(kp[1]))

    # 计算四条边的长度
    edges = {
        'rear':  (pts['rear_left'],  pts['rear_right']),
        'front': (pts['front_left'], pts['front_right']),
        'left':  (pts['rear_left'],  pts['front_left']),
        'right': (pts['rear_right'], pts['front_right']),
    }

    edge_lens = {k: edge_length(p1, p2) for k, (p1, p2) in edges.items()}

    # 短边 = rear 和 front（箱子前后边通常比左右边短）
    # 但也可能是 left 和 right，取实际较短的两条
    sorted_edges = sorted(edge_lens.items(), key=lambda x: x[1])
    short_edge_1_name, short_edge_2_name = sorted_edges[0][0], sorted_edges[1][0]

    # 计算短边中点
    midpoints = {}
    for edge_name in [short_edge_1_name, short_edge_2_name]:
        p1, p2 = edges[edge_name]
        mid_u = (p1[0] + p2[0]) / 2.0
        mid_v = (p1[1] + p2[1]) / 2.0
        midpoints[edge_name] = (mid_u, mid_v)

    # 读取深度值（双线性插值）
    result_row = {
        'frame': color_path.stem,
        'box_conf': float(r.boxes.conf[0]) if r.boxes is not None and len(r.boxes.conf) > 0 else 0,
    }

    for edge_name in [short_edge_1_name, short_edge_2_name]:
        mu, mv = midpoints[edge_name]

        # 深度插值（处理亚像素）
        u0, v0 = int(np.floor(mu)), int(np.floor(mv))
        u1, v1 = min(u0+1, IMG_W-1), min(v0+1, IMG_H-1)
        du, dv = mu - u0, mv - v0

        d00 = depth[v0, u0]
        d10 = depth[v0, u1]
        d01 = depth[v1, u0]
        d11 = depth[v1, u1]

        # 忽略无效深度 (0)
        valid_depths = [d for d in [d00, d10, d01, d11] if d > 0]
        if not valid_depths:
            depth_mm = 0
        else:
            # 双线性插值（只对有效深度）
            d00 = d00 if d00 > 0 else np.mean(valid_depths)
            d10 = d10 if d10 > 0 else np.mean(valid_depths)
            d01 = d01 if d01 > 0 else np.mean(valid_depths)
            d11 = d11 if d11 > 0 else np.mean(valid_depths)
            depth_mm = (1-du)*(1-dv)*d00 + du*(1-dv)*d10 + (1-du)*dv*d01 + du*dv*d11

        # → 3D
        p3d = pixel_to_3d(mu, mv, depth_mm)

        result_row[f'{edge_name}_mid_u'] = mu
        result_row[f'{edge_name}_mid_v'] = mv
        result_row[f'{edge_name}_depth_mm'] = depth_mm
        result_row[f'{edge_name}_X_mm'] = p3d[0]
        result_row[f'{edge_name}_Y_mm'] = p3d[1]
        result_row[f'{edge_name}_Z_mm'] = p3d[2]

    # 记录四条边长度
    for k, v in edge_lens.items():
        result_row[f'{k}_len_px'] = v

    # 4个角点坐标
    for name in names:
        result_row[f'{name}_u'] = pts[name][0]
        result_row[f'{name}_v'] = pts[name][1]

    # --- 绘制可视化 ---
    if vis_dir is not None:
        vis = color.copy()
        h, w = vis.shape[:2]

        # 边界框
        if r.boxes is not None:
            for box in r.boxes:
                if box.conf[0] >= 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 四条边 + 关键点
        pts_list = [pts[n] for n in names]
        for idx, (i, j) in enumerate(EDGES):
            p1 = (int(pts_list[i][0]), int(pts_list[i][1]))
            p2 = (int(pts_list[j][0]), int(pts_list[j][1]))
            ename = EDGE_NAMES[idx]
            # 短边用红色粗线，长边用蓝色细线
            is_short = (ename == short_edge_1_name or ename == short_edge_2_name)
            color = (0, 0, 255) if is_short else (255, 0, 0)
            thickness = 3 if is_short else 1
            cv2.line(vis, p1, p2, color, thickness)
            # 边长标注
            mid_lbl = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            cv2.putText(vis, f"{ename}({edge_lens[ename]:.0f}px){'*' if is_short else ''}",
                       mid_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 四个角点
        for i, name in enumerate(names):
            px, py = int(pts[name][0]), int(pts[name][1])
            cv2.circle(vis, (px, py), 6, KP_COLORS[i], -1)
            cv2.putText(vis, name, (px+10, py-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, KP_COLORS[i], 1)

        # 短边中点（大圆 + 3D坐标）
        for edge_name in [short_edge_1_name, short_edge_2_name]:
            mu, mv = midpoints[edge_name]
            mx, my = int(mu), int(mv)
            cv2.circle(vis, (mx, my), 8, (0, 255, 255), -1)
            cv2.circle(vis, (mx, my), 10, (0, 255, 255), 2)
            d = result_row.get(f'{edge_name}_depth_mm', 0)
            x3 = result_row.get(f'{edge_name}_X_mm', 0)
            y3 = result_row.get(f'{edge_name}_Y_mm', 0)
            z3 = result_row.get(f'{edge_name}_Z_mm', 0)
            cv2.putText(vis, f"{edge_name}_mid", (mx+14, my-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(vis, f"3D:({float(x3):.0f},{float(y3):.0f},{float(z3):.0f})mm",
                       (mx+14, my+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 帧号
        cv2.putText(vis, f"{color_path.stem} | conf:{result_row['box_conf']:.2f}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(str(vis_dir / f'{color_path.stem}_annotated.jpg'), vis)

    return result_row


def main():
    print(f"加载模型: test5 best.pt")
    model_path = "E:/Projects/Datasets/box_grasp/single_test5/runs/pose/runs/test5/train/weights/best.pt"
    model = YOLO(model_path)

    vis_dir = OUTPUT_DIR / 'annotated'
    vis_dir.mkdir(exist_ok=True)

    color_files = sorted(COLOR_DIR.glob('*.png'))
    print(f"共 {len(color_files)} 帧")

    all_results = []
    skip_count = 0

    for i, cf in enumerate(color_files):
        df_path = DEPTH_DIR / cf.name
        if not df_path.exists():
            skip_count += 1
            continue

        row = process_frame(cf, df_path, model, vis_dir)
        if row is None:
            skip_count += 1
            continue

        all_results.append(row)

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(color_files)}")

    print(f"\n处理完成: {len(all_results)} 帧成功, {skip_count} 帧跳过")

    # 保存结果（收集所有列名，处理不同帧短边名不同的情况）
    csv_path = OUTPUT_DIR / 'results_3d.csv'
    if all_results:
        all_keys = set()
        for row in all_results:
            all_keys.update(row.keys())
        fieldnames = ['frame', 'box_conf'] + sorted(k for k in all_keys if k not in ['frame', 'box_conf'])
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
    print(f"结果保存: {csv_path}")

    # 统计：汇总四条边的中点3D坐标（每条边在其作为短边时记录）
    if len(all_results) > 0:
        # 统计各边边长
        for edge_name in ['rear', 'front', 'left', 'right']:
            lens = [r[f'{edge_name}_len_px'] for r in all_results]
            print(f"  {edge_name} 边长: {np.mean(lens):.1f} ± {np.std(lens):.1f} px")

        # 找出最常作为短边的两条边，汇总3D坐标
        edge_3d = {e: {'X': [], 'Y': [], 'Z': []} for e in ['rear', 'front', 'left', 'right']}
        for r in all_results:
            for e in ['rear', 'front', 'left', 'right']:
                if f'{e}_X_mm' in r:
                    edge_3d[e]['X'].append(r[f'{e}_X_mm'])
                    edge_3d[e]['Y'].append(r[f'{e}_Y_mm'])
                    edge_3d[e]['Z'].append(r[f'{e}_Z_mm'])

        # 实际每帧的短边是哪两条
        short_counts = {}
        for r in all_results:
            for k in r:
                if k.endswith('_X_mm'):
                    e = k.replace('_X_mm', '')
                    short_counts[e] = short_counts.get(e, 0) + 1
        print(f"\n各边作为短边出现次数: {short_counts}")

        for e, data in edge_3d.items():
            xs = [float(v) for v in data['X']]
            ys = [float(v) for v in data['Y']]
            zs = [float(v) for v in data['Z'] if float(v) > 0]
            if zs:
                print(f"\n{e} 短边中点 3D (相机系, mm), {len(zs)}帧:")
                print(f"  X: {np.mean(xs):.1f} ± {np.std(xs):.1f}")
                print(f"  Y: {np.mean(ys):.1f} ± {np.std(ys):.1f}")
                print(f"  Z: {np.mean(zs):.1f} ± {np.std(zs):.1f}")


if __name__ == '__main__':
    main()
