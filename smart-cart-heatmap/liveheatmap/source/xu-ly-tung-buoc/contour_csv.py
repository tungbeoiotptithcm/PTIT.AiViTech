import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
from ultralytics import YOLO
from scipy.stats import gaussian_kde
import csv

# === Cấu hình video mô phỏng ===
VIDEO_PATH = "people2.mp4"  # Đường dẫn tới video
cap = cv2.VideoCapture(VIDEO_PATH)

# === Cấu hình detector người YOLOv8 ===
model = YOLO("yolov8s.pt")  # Dùng mô hình YOLOv8s
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# === Tập dữ liệu để vẽ heatmap và lưu tọa độ ===
heatmap_data = []
all_coordinates = []  # Danh sách để lưu tọa độ của tất cả các frame

frame_skip = 29  # xử lý mỗi frame
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

    frame_coordinates = []  # Danh sách để lưu tọa độ của frame hiện tại
    for result in results:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            conf = confs[i]
            cls = int(clss[i])
            if cls == 0:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                heatmap_data.append((cx, cy))
                frame_coordinates.append((cx, cy))  # Lưu tọa độ của người trong frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    all_coordinates.append(frame_coordinates)  # Lưu tọa độ của frame vào danh sách chung

    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Vẽ heatmap và contour ===
heatmap_array = np.zeros((480, 640))
for (x, y) in heatmap_data:
    if 0 <= y < 480 and 0 <= x < 640:
        heatmap_array[y, x] += 1

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(heatmap_array, cmap="jet", cbar=True, vmax=5)
plt.title("Heatmap chuyển động người")
plt.gca().invert_yaxis()
plt.xticks(np.arange(0, 641, 50))
plt.yticks(np.arange(0, 481, 50))
#plt.xlabel("Ox")
#plt.ylabel("Oy")
plt.gca().set_xticks(np.arange(0, 641, 50))
plt.gca().set_yticks(np.arange(0, 481, 50))

plt.subplot(1, 2, 2)
# === Vẽ contour dựa trên heatmap_array ===
# Tạo lưới đều trên khung hình
xgrid = np.linspace(0, 640, 640)
ygrid = np.linspace(0, 480, 480)
X, Y = np.meshgrid(xgrid, ygrid)

# Ước lượng mật độ hạt nhân từ heatmap_data
kde = gaussian_kde(np.array(heatmap_data).T)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# Vẽ contour dạng filled
cf = plt.contourf(X, Y, Z, levels=20, cmap='jet')
plt.colorbar(cf, label="Mật độ")

# Tùy chỉnh
plt.title("Mật độ chuyển động người (Contour)")
#plt.xlabel("Ox")
#plt.ylabel("Oy")
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.xlim(0, 640)
plt.ylim(0, 480)
plt.xticks(np.arange(0, 641, 100))
plt.yticks(np.arange(0, 481, 50))

# Ghi chú khu vực
region_names = {
    (100, 100): "Chân ga",
    (200, 100): "Thảm lau",
    (300, 100): "Vải",
    (400, 100): "Áo",
    (500, 100): "Quần",
    (600, 100): "Rèm",

    (100, 200): "Thịt",
    (200, 200): "Cá",
    (300, 200): "Rau củ",
    (400, 200): "Hải sản",
    (500, 200): "Đồ khô",
    (600, 200): "Trái cây",

    (100, 300): "Chén",
    (200, 300): "Nồi",
    (300, 300): "Chảo",
    (400, 300): "Kệ",
    (500, 300): "Đũa",
    (600, 300): "Bếp",

    (100, 400): "Bánh kẹo",
    (200, 400): "Trà",
    (300, 400): "Nước ngọt",
    (400, 400): "Gia vị",
    (500, 400): "Quà",
    (600, 400): "Đồ ăn vặt",
}

for (x_reg, y_reg), name in region_names.items():
    plt.text(x_reg, y_reg, name, fontsize=6, ha='center', va='center', color='white')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.tight_layout()
plt.savefig(f"heatmap_contour_{timestamp}.png")
plt.show()

# === Xuất tọa độ ra file CSV ===
csv_filename = f"person_coordinates_{timestamp}.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Frame', 'Person', 'X', 'Y'])  # Viết header

    for frame_number, frame_coords in enumerate(all_coordinates):
        for person_number, (x, y) in enumerate(frame_coords):
            csv_writer.writerow([frame_number + 1, person_number + 1, x, y])  # Viết dữ liệu

print(f"Tọa độ đã được lưu vào file: {csv_filename}")
# test file rule_from
