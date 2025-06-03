import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import cv2
import torch
from ultralytics import YOLO
from scipy.stats import gaussian_kde, percentileofscore
import threading
from tkinter import messagebox, filedialog

class GridMapApp:
    def create_heatmap(self):
        # Kiểm tra xem bản đồ đã được tạo chưa
        if self.rows is None or self.cols is None:
            messagebox.showerror("Lỗi", "Vui lòng tạo bản đồ trước khi tạo heatmap.")
            return

        # Mở hộp thoại để chọn file video
        video_path = filedialog.askopenfilename(
            title="Chọn video đầu vào",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")],
            initialdir="D:/IOT CHALLENGE/smart-cart-heatmap/liveheatmap/data/video"
        )

        # Kiểm tra nếu không chọn file video
        if not video_path:
            messagebox.showinfo("Thông báo", "Không có file video nào được chọn.")
            return

        def worker():
            try:
                # Thiết lập lề và mở file video
                margin = getattr(self, 'margin', 40)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    messagebox.showerror("Lỗi", "Không thể mở file video.")
                    return

                # Tải mô hình YOLO và chọn thiết bị xử lý
                model = YOLO("yolov8n.pt")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model.to(device)

                # Khởi tạo dữ liệu heatmap và tọa độ
                heatmap_data = []
                all_coordinates = []
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_skip = int(fps) if fps > 0 else 30
                frame_count = 0

                # Hiển thị thông báo đang xử lý
                self.master.after(0, lambda: messagebox.showinfo("Đang xử lý", "Đang tạo heatmap, vui lòng chờ..."))

                # Xử lý từng khung hình trong video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_coordinates = []
                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue

                    # Chuyển đổi và dự đoán đối tượng trong khung hình
                    frame = cv2.resize(frame, (640, 480))
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model.predict(rgb_frame, verbose=False)

                    # Xử lý kết quả dự đoán để lấy tọa độ người
                    for result in results:
                        boxes = result.boxes
                        xyxy = boxes.xyxy.cpu().numpy()
                        clss = boxes.cls.cpu().numpy()
                        for i in range(len(xyxy)):
                            x1, y1, x2, y2 = map(int, xyxy[i])
                            if int(clss[i]) == 0:  # Người
                                cx = (x1 + x2) // 2
                                cy = (y1 + y2) // 2
                                heatmap_data.append((cx, cy))
                                frame_coordinates.append((cx, cy))
                                # Hiển thị khung màu xanh
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Lưu tọa độ đã điều chỉnh lề
                    frame_coordinates = []
                    for result in results:
                        boxes = result.boxes
                        xyxy = boxes.xyxy.cpu().numpy()
                        clss = boxes.cls.cpu().numpy()
                        for i in range(len(xyxy)):
                            x1, y1, x2, y2 = map(int, xyxy[i])
                            if int(clss[i]) == 0:
                                cx = (x1 + x2) // 2 - margin
                                cy = (y1 + y2) // 2 - margin
                                heatmap_data.append((cx, cy))
                                frame_coordinates.append((cx, cy))
                    all_coordinates.append(frame_coordinates)

                    # Hiển thị khung hình với khung bao quanh người
                    cv2.imshow("Tracking Person", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Giải phóng tài nguyên video
                cap.release()
                cv2.destroyAllWindows()

                # Tạo heatmap nếu có dữ liệu
                if len(heatmap_data) > 0:
                    # Chuẩn bị lưới tọa độ cho heatmap
                    points = np.array(heatmap_data).T
                    xgrid = np.linspace(0, 640 - 2*margin, 640 - 2*margin)
                    ygrid = np.linspace(0, 480 - 2*margin, 480 - 2*margin)
                    X, Y = np.meshgrid(xgrid, ygrid)
                    kde = gaussian_kde(points, bw_method=0.2)
                    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

                    # Tạo bản đồ nền và heatmap
                    map_img = self.get_map_image().resize((640, 480))
                    map_array = np.array(map_img)
                    margin = self.margin
                    map_img_full = self.get_map_image()
                    width, height = map_img_full.size
                    cropped_img = map_img_full.crop((margin, margin, width, height))
                    map_img = cropped_img.resize((640 - 2*margin, 480 - 2*margin))

                    # Vẽ heatmap lên bản đồ
                    plt.figure(figsize=(10, 8))
                    plt.imshow(map_img, alpha=0.9, extent=[margin, 640 - margin, 480 - margin, margin])
                    plt.imshow(Z, extent=[margin, 640 - margin, 480 - margin, margin], cmap='jet', alpha=0.4)
                    plt.title("Heatmap & Contour mật độ người trên bản đồ siêu thị", fontsize=14)
                    plt.axis('off')
                    plt.tight_layout()

                    # Lưu heatmap
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"D:/IOT CHALLENGE/smart-cart-heatmap/liveheatmap/data/1.heatmap_contour_folder/heatmap_contour_{timestamp}.png"
                    plt.savefig(fname, bbox_inches='tight', dpi=150)
                    plt.close()
                    self.master.after(0, lambda: messagebox.showinfo("Thành công", f"Đã lưu heatmap tại: {fname}"))

                    # Phân tích và tạo gợi ý tối ưu hóa
                    region_info = []
                    regions = self.find_connected_regions()
                    map_img_full = self.get_map_image()
                    width, height = map_img_full.size
                    scaled_width = 640 - 2 * margin
                    scaled_height = 480 - 2 * margin
                    scale_x = scaled_width / (width - margin)
                    scale_y = scaled_height / (height - margin)
                    all_densities = Z.ravel()

                    # Tính mật độ và gợi ý cho từng khu vực
                    for loai, cells in regions:
                        if loai != "Đường đi":
                            xs = [c * self.cell_size + self.margin + self.cell_size // 2 for _, c in cells]
                            ys = [r * self.cell_size + self.margin + self.cell_size // 2 for r, _ in cells]
                            cx = int(sum(xs) / len(xs))
                            cy = int(sum(ys) / len(ys))
                            x_scaled = int((cx - margin) * scale_x)
                            y_scaled = int((cy - margin) * scale_y)

                            if 0 <= x_scaled < Z.shape[1] and 0 <= y_scaled < Z.shape[0]:
                                density = np.mean(Z[max(0, y_scaled - 5):min(Z.shape[0], y_scaled + 6),
                                                    max(0, x_scaled - 5):min(Z.shape[1], x_scaled + 6)])
                                percentile = percentileofscore(all_densities, density)

                                # Đưa ra gợi ý dựa trên mật độ
                                if percentile >= 75:
                                    suggestion = "Mở rộng khu vực hoặc thêm quầy phục vụ"
                                elif percentile >= 50:
                                    suggestion = "Tối ưu bố trí sản phẩm"
                                elif percentile >= 25:
                                    suggestion = "Duy trì hiện trạng"
                                else:
                                    suggestion = "Cân nhắc giảm diện tích"

                                region_info.append({
                                    'Khu vực': f"Khu {loai.lower()}",
                                    'Tọa độ X': x_scaled,
                                    'Tọa độ Y': y_scaled,
                                    'Mật độ': density,
                                    'Phần trăm': f"{percentile:.1f}%",
                                    'Đề xuất': suggestion
                                })

                    # Tạo DataFrame từ thông tin vùng
                    df = pd.DataFrame(region_info)

                    # Tạo ảnh gợi ý
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(map_img, extent=[0, Z.shape[1], Z.shape[0], 0], alpha=1.0)
                    scatter = ax.scatter(
                        df['Tọa độ X'],
                        df['Tọa độ Y'],
                        c=df['Mật độ'],
                        cmap='coolwarm',
                        s=200,
                        edgecolors='black'
                    )

                    # Thêm thanh màu cho mật độ
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Mật độ')

                    # Hiển thị nhãn và gợi ý trên ảnh
                    for i, row in df.iterrows():
                        ax.text(row['Tọa độ X'], row['Tọa độ Y'] - 10, f"{row['Khu vực']}", ha='center',
                                fontsize=10, fontweight='bold', color='black')
                        ax.text(row['Tọa độ X'], row['Tọa độ Y'] + 20, f"{row['Đề xuất']}", ha='center',
                                fontsize=8, color='gray')

                    # Cài đặt tiêu đề và lưu ảnh gợi ý
                    ax.set_title('Store Suggestion Map')
                    ax.set_xlim(0, Z.shape[1])
                    ax.set_ylim(Z.shape[0], 0)
                    ax.axis('off')
                    plt.tight_layout()
                    suggestion_img_filename = f"D:/IOT CHALLENGE/smart-cart-heatmap/liveheatmap/data/2.store_suggestion_map/store_suggestion_map_{timestamp}.png"
                    plt.savefig(suggestion_img_filename, dpi=150)
                    plt.close(fig)

                    # Xuất file CSV gợi ý
                    csv_suggestion_filename = f"D:/IOT CHALLENGE/smart-cart-heatmap/liveheatmap/data/4.store_optimization_folder/store_optimization_{timestamp}.csv"
                    df.to_csv(csv_suggestion_filename, index=False)

                    # Hiển thị thông báo gợi ý
                    self.master.after(0, lambda: messagebox.showinfo("Gợi ý tối ưu",
                        f"Đã lưu gợi ý tại: {suggestion_img_filename}\n\n" +
                        "Các đề xuất chính:\n" +
                        "\n".join([f"- {row['Khu vực']}: {row['Đề xuất']}" for i, row in df.iterrows()])
                    ))

                else:
                    # Thông báo nếu không có dữ liệu heatmap
                    self.master.after(0, lambda: messagebox.showwarning("Không có dữ liệu", "Không có dữ liệu heatmap để vẽ contour."))

                # Xuất tọa độ ra file CSV
                if all_coordinates:
                    flat_coords = []
                    for frame_number, frame_coords in enumerate(all_coordinates):
                        for person_number, (x, y) in enumerate(frame_coords):
                            flat_coords.append([frame_number + 1, person_number + 1, x, y])

                    df_coords = pd.DataFrame(flat_coords, columns=['Frame', 'Person', 'X', 'Y'])
                    csv_filename = f"D:/IOT CHALLENGE/smart-cart-heatmap/liveheatmap/data/3.person_coordinates_folder/person_coordinates_{timestamp}.csv"
                    df_coords.to_csv(csv_filename, index=False)
                    self.master.after(0, lambda: messagebox.showinfo("Thành công", f"Tọa độ đã được lưu vào file: {csv_filename}"))

            except Exception as e:
                # Hiển thị lỗi nếu có
                self.master.after(0, lambda e=e: messagebox.showerror("Lỗi", f"Không thể tạo heatmap: {str(e)}"))

        # Chạy xử lý trong luồng riêng
        threading.Thread(target=worker, daemon=True).start()