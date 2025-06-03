import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
from ultralytics import YOLO
from scipy.stats import gaussian_kde
import re
import csv

# === Cấu hình video mô phỏng ===
VIDEO_PATH = "people3.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# === Cấu hình detector người YOLOv8 ===
model = YOLO("yolov8s.pt")  # Hoặc thử yolov8n.pt hoặc yolov8nano.pt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# === Tập dữ liệu để vẽ heatmap và lưu tọa độ ===
heatmap_data = []
all_coordinates = []

frame_skip = 29 #số frame bỏ đi mỗi giây/ không tính (fps của video - 1)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện người bằng YOLOv8
    results = model.predict(rgb_frame, verbose=False)

    frame_coordinates = []
    for result in results:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            if int(clss[i]) == 0:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                heatmap_data.append((cx, cy))
                frame_coordinates.append((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bỏ nếu không cần vẽ

    all_coordinates.append(frame_coordinates)

    cv2.imshow("YOLOv8 Detection", frame)  # Bỏ nếu không cần hiển thị
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Vẽ heatmap và contour ===
heatmap_array = np.zeros((480, 640), dtype=np.float32)
if heatmap_data:
    points = np.array(heatmap_data).T
    np.add.at(heatmap_array, (points[1], points[0]), 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(heatmap_array, cmap="jet", cbar=True, vmax=5)
plt.title("Heatmap")
#plt.gca().invert_yaxis()
plt.xticks(np.arange(0, 641, 50))
plt.yticks(np.arange(0, 481, 50))
plt.gca().set_xticks(np.arange(0, 641, 50))
plt.gca().set_yticks(np.arange(0, 481, 50))

plt.subplot(1, 2, 2)
# === Vẽ contour dựa trên heatmap_array ===
# Tạo lưới đều trên khung hình
xgrid = np.linspace(0, 640, 640)
ygrid = np.linspace(0, 480, 480)
X, Y = np.meshgrid(xgrid, ygrid)

# Ước lượng mật độ hạt nhân từ heatmap_data
if heatmap_data:
    points = np.array(heatmap_data).T
    if points.shape[1] > 10000:  # Lấy mẫu nếu có quá nhiều điểm
        sample_indices = np.random.choice(points.shape[1], 10000, replace=False)
        points = points[:, sample_indices]
    kde = gaussian_kde(points, bw_method=0.2)  # Tinh chỉnh bandwidth
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Vẽ contour dạng filled
    cf = plt.contourf(X, Y, Z, levels=20, cmap='jet')
    plt.colorbar(cf, label="Mật độ")

    # Tùy chỉnh
    plt.title("Contour")
    #plt.gca().invert_yaxis()
    plt.xlim(0, 640)
    plt.ylim(0, 480)

    #plt.gca().invert_yaxis()

    plt.xticks(np.arange(0, 641, 100))
    plt.yticks(np.arange(0, 481, 50))

    # Ghi chú khu vực
    region_names = {
        (100, 100): "Chăn ga", (200, 100): "Thảm lau", (300, 100): "Vải", (400, 100): "Áo",
        (500, 100): "Quần", (600, 100): "Rèm", (100, 200): "Thịt", (200, 200): "Cá",
        (300, 200): "Rau củ", (400, 200): "Hải Sản", (500, 200): "Đồ Khô", (600, 200): "Trái cây",
        (100, 300): "Chén", (200, 300): "Nồi", (300, 300): "Chảo", (400, 300): "Kệ",
        (500, 300): "Đũa", (600, 300): "Bếp", (100, 400): "Bánh kẹo", (200, 400): "Trà",
        (300, 400): "Nước ngọt", (400, 400): "Gia vị", (500, 400): "Quà", (600, 400): "Đồ ăn vặt",
    }
    for (x_reg, y_reg), name in region_names.items():
        plt.text(x_reg, y_reg, name, fontsize=6, ha='center', va='center', color='white')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.tight_layout()
plt.savefig(f"D:\IOT CHALLENGE\smart-cart-heatmap\liveheatmap\data\heatmap_contour_folder\heatmap_contour_{timestamp}.png")
plt.close()

print(f"Đã lưu heatmap và contour tại: heatmap_contour_{timestamp}.png")

# === Bước 4: Tính toán mật độ và lưu trữ thông tin cho trực quan hóa ===
region_info = []
if heatmap_data:
    for (X, Y), name in region_names.items():
        if 0 <= X < 640 and 0 <= Y < 480:
            density = Z[int(Y), int(X)]
            percentile_rank = (np.sum(Z <= density) / Z.size) * 100
            suggestion = ""
            color = "white"
            if percentile_rank >= 80:
                suggestion = "Cần tăng cường"; color = "red"
            elif percentile_rank >= 60:
                suggestion = "Nên bổ sung"; color = "orange"
            elif percentile_rank >= 40:
                suggestion = "Theo dõi"; color = "yellow"
            elif percentile_rank >= 20:
                suggestion = "Điều phối"; color = "blue"
            else:
                suggestion = "Cân nhắc giảm"; color = "green"
            region_info.append({'name': name, 'density': density, 'percentile': percentile_rank, 'suggestion': suggestion, 'color': color})

    # === Bước 5: Sắp xếp dữ liệu theo hạng phần trăm từ cao xuống thấp ===
    region_info_sorted = sorted(region_info, key=lambda item: item['percentile'], reverse=True)

    # === Bước 6: Tạo hình ảnh kết hợp biểu đồ cột và text ===
    fig_bar_text, (ax1_bar, ax2_text) = plt.subplots(1, 2, figsize=(18, 8))

    # --- Biểu đồ cột (nửa bên trái) ---
    bar_names = [info['name'] for info in region_info_sorted]
    percentiles = [info['percentile'] for info in region_info_sorted]
    colors = [info['color'] for info in region_info_sorted]

    ax1_bar.bar(bar_names, percentiles, color=colors)
    ax1_bar.set_xlabel('Khu vực', fontsize=12)
    ax1_bar.set_ylabel('Hạng phần trăm mật độ', fontsize=12)
    ax1_bar.set_title('Mật độ người theo khu vực', fontsize=14)
    ax1_bar.tick_params(axis='y', labelsize=10)
    ax1_bar.set_xticklabels(bar_names, rotation=45, ha="right", fontsize=10)

    # --- Text gợi ý (nửa bên phải) ---
    text_output = "=== GỢI Ý ĐIỀU CHỈNH CỬA HÀNG ===\n"
    for info in region_info_sorted:
        text_output += f"- Khu vực {info['name']} ({info['percentile']:.2f}%): {info['suggestion']}\n"

    ax2_text.text(0.05, 0.95, text_output, transform=ax2_text.transAxes, fontsize=15, verticalalignment='top')
    ax2_text.axis('off')

    plt.tight_layout()

    # Lưu hình ảnh biểu đồ cột và text
    output_filename_bar_text = f"D:\IOT CHALLENGE\smart-cart-heatmap\liveheatmap\data\graph_bar_recommendations_folder\cot_goi_y_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_filename_bar_text)
    plt.close(fig_bar_text)

    print(f"Đã lưu hình ảnh kết hợp biểu đồ cột và text tại: {output_filename_bar_text}")

# === Xuất tọa độ ra file CSV ===
if all_coordinates:
    flat_coords = []
    for frame_number, frame_coords in enumerate(all_coordinates):
        for person_number, (x, y) in enumerate(frame_coords):
            flat_coords.append([frame_number + 1, person_number + 1, x, y])

    df_coords = pd.DataFrame(flat_coords, columns=['Frame', 'Person', 'X', 'Y'])
    csv_filename = f"D:\IOT CHALLENGE\smart-cart-heatmap\liveheatmap\data\person_coordinates_folder\person_coordinates_{timestamp}.csv"
    df_coords.to_csv(csv_filename, index=False)
    print(f"Tọa độ đã được lưu vào file: {csv_filename}")

#run_time_frame1: 3p30
#run_time_frame5: 45s