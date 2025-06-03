import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
from ultralytics import YOLO
from scipy.stats import gaussian_kde

# === Cấu hình video mô phỏng ===
VIDEO_PATH = "people2.mp4"  # Đường dẫn tới video
cap = cv2.VideoCapture(VIDEO_PATH)

# === Cấu hình detector người YOLOv8 ===
model = YOLO("yolov8s.pt")  # Dùng mô hình YOLOv8s
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# === Tập dữ liệu để vẽ heatmap ===
heatmap_data = []

frame_skip = 1  # xử lý mỗi frame
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
plt.xticks(np.arange(0, 641, 50))
plt.yticks(np.arange(0, 481, 50))

# Ghi chú khu vực
region_names = {
    (550, 150): "Lối ra",
    (200, 150): "Thức Uống",
    (200, 200): "Thu Ngân",
    (350, 200): "Bánh Kẹo",
    (300, 400): "Gia Dụng",
    (150, 300): "Quần Áo"
}
for (x_reg, y_reg), name in region_names.items():
    plt.text(x_reg, y_reg, name, fontsize=9, ha='center', va='center', color='white')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.tight_layout()
plt.savefig(f"heatmap_contour_{timestamp}.png")
plt.show()