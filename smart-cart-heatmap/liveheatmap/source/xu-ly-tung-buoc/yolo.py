import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
import torch

# === Cấu hình video mô phỏng ===
VIDEO_PATH = "people.mp4"  # Đường dẫn tới video của bạn
cap = cv2.VideoCapture(VIDEO_PATH)

# === Cấu hình detector người YOLOv8 ===
model = torch.hub.load('ultralytics/yolov8', 'yolov8s', pretrained=True)

# Đặt device là 'cuda' nếu bạn có GPU, ngược lại để 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# === Tập dữ liệu để vẽ heatmap và clustering ===
heatmap_data = []
clustering_data = []

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # YOLOv8 thường làm việc với ảnh RGB

    # Phát hiện người bằng YOLOv8
    results = model.predict(rgb_frame, verbose=False)

    # Lấy bounding boxes của những người được phát hiện
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4]
            cls = int(box[5])
            if cls == 0:  # 0 là lớp "person" trong COCO
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                heatmap_data.append((cx, cy))
                clustering_data.append([cx, cy])  # Thêm tọa độ vào dữ liệu clustering
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Hiển thị khung hình trong quá trình chạy (có thể tắt)
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Vẽ heatmap ===
heatmap_array = np.zeros((480, 640))
for (x, y) in heatmap_data:
    if 0 <= y < 480 and 0 <= x < 640:
        heatmap_array[y, x] += 1

plt.figure(figsize=(12, 6))  # Tăng kích thước figure để vẽ cả heatmap và clustering

# Subplot cho Heatmap
plt.subplot(1, 2, 1)  # Tạo subplot cho heatmap (1 hàng, 2 cột, vị trí 1)
sns.heatmap(heatmap_array, cmap="jet", cbar=True, vmax=3)
plt.title("Heatmap chuyển động người")
plt.gca().invert_yaxis()
plt.xticks(np.arange(0, 641, 50))  # Chia nhỏ trục Ox cách nhau 50 đơn vị
plt.yticks(np.arange(0, 481, 50))  # Chia nhỏ trục Oy cách nhau 50 đơn vị
plt.xlabel("Ox")
plt.ylabel("Oy")
plt.gca().set_xticks(np.arange(0, 641, 50))  # Thiết lập lại ticks Ox để hiển thị nhãn
plt.gca().set_yticks(np.arange(0, 481, 50))  # Thiết lập lại ticks Oy để hiển thị nhãn

# === Clustering ===
if clustering_data:
    n_clusters = 5  # Số lượng cluster bạn muốn tìm
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(clustering_data)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Vẽ kết quả clustering lên subplot thứ hai
    plt.subplot(1, 2, 2)  # Tạo subplot cho clustering (1 hàng, 2 cột, vị trí 2)
    plt.scatter([point[0] for point in clustering_data], [point[1] for point in clustering_data], c=clusters, cmap='viridis', s=20, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(f"K-Means Clustering ({n_clusters} clusters)")
    plt.xlabel("Ox")
    plt.ylabel("Oy")
    plt.gca().invert_yaxis()  # Đảo ngược trục Oy cho phần clustering
    plt.gca().invert_xaxis()  # Đảo ngược trục Ox cho phần clustering
    plt.xlim(0, 600)
    plt.ylim(0, 450)
    plt.xticks(np.arange(0, 601, 50))  # Chia nhỏ trục Ox cách nhau 50 đơn vị
    plt.yticks(np.arange(0, 451, 50))  # Chia nhỏ trục Oy cách nhau 50 đơn vị
    plt.legend()

    # === Nhập tọa độ và tên khu vực TRONG CODE ===
    region_names = {
        (550, 150): "Lối ra",
        (200, 150): "Thức Uống",
        (200, 200): "Thu Ngân",
        (350, 200): "Bánh Kẹo",
        (300, 400): "Gia Dụng",
        (150, 300): "Quần Áo"
    }
    # Thêm các tọa độ và tên khu vực khác vào dictionary này

    # Chèn tên khu vực vào biểu đồ
    for (x_reg, y_reg), name in region_names.items():
        plt.text(x_reg, y_reg, name, fontsize=9, ha='center', va='center', color='black')

else:
    print("Không có dữ liệu để thực hiện clustering.")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.tight_layout()  # Điều chỉnh layout để các subplot không bị chồng chéo
plt.savefig(f"heatmap_clustering_with_yolov8_{timestamp}.png")
plt.show()