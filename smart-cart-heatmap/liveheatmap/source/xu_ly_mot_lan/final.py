import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import cv2
import torch
from ultralytics import YOLO
from scipy.stats import gaussian_kde, percentileofscore
from matplotlib.colors import LogNorm
import threading


class GridMapApp:
    def __init__(self, master):
        # Khởi tạo cửa sổ ứng dụng chính và các thuộc tính cơ bản
        self.master = master
        self.master.title("Grid Map")
        self.rows = None
        self.cols = None
        self.rect_types = {}

        # Định nghĩa ánh xạ màu cho các loại ô lưới
        self.type_colors = {
            "Cửa ra vào": "red",
            "Đồ uống": "lightblue",
            "Gia vị": "orange",
            "Nhu yếu phẩm": "lightgreen",
            "Lương thực": "khaki",
            "Bánh kẹo": "pink",
            "Thu ngân": "brown",
            "Đường đi": "#eeeeee"
        }

        # Tạo khung nhập liệu cho số hàng, cột và nút mở file
        self.input_frame = tk.Frame(master)
        self.input_frame.pack(pady=10)

        # Tạo các nhãn và ô nhập liệu cho số hàng và cột
        tk.Label(self.input_frame, text="Số hàng (2-99):").grid(row=0, column=0, padx=5)
        self.entry_rows = tk.Entry(self.input_frame, width=7)
        self.entry_rows.grid(row=0, column=1, padx=5)
        tk.Label(self.input_frame, text="Số cột (2-99):").grid(row=0, column=2, padx=5)
        self.entry_cols = tk.Entry(self.input_frame, width=7)
        self.entry_cols.grid(row=0, column=3, padx=5)

        # Tạo các nút chức năng: tạo bản đồ, mở file CSV
        self.btn_create = tk.Button(self.input_frame, text="Tạo map mới", command=self.create_new_map)
        self.btn_create.grid(row=0, column=4, padx=8)
        self.btn_open = tk.Button(self.input_frame, text="Mở file CSV", command=self.open_csv)
        self.btn_open.grid(row=0, column=5, padx=8)

        # Tạo các nút xuất file và tạo heatmap
        self.btn_export_png = tk.Button(self.input_frame, text="Xuất file PNG", command=self.export_image)
        self.btn_export_png.grid(row=0, column=6, padx=5)
        self.btn_export_csv = tk.Button(self.input_frame, text="Xuất file CSV", command=self.export_csv)
        self.btn_export_csv.grid(row=0, column=7, padx=5)
        self.btn_heatmap = tk.Button(self.input_frame, text="Tạo Heatmap", command=self.create_heatmap)
        self.btn_heatmap.grid(row=0, column=8, padx=5)

        # Khởi tạo các biến canvas và khung nút
        self.canvas = None
        self.btn_frame = None

    def create_new_map(self):
        # Kiểm tra và lấy số hàng, cột từ ô nhập liệu
        try:
            rows = int(self.entry_rows.get())
            cols = int(self.entry_cols.get())
            if rows < 2 or rows > 99 or cols < 2 or cols > 99:
                raise ValueError
        except:
            messagebox.showerror("Lỗi", "Số hàng và số cột phải là số nguyên từ 2 đến 99.")
            return

        # Lưu số hàng, cột và khởi tạo bản đồ mới
        self.rows = rows
        self.cols = cols
        self.rect_types = {}
        self.show_map_ui()

    def open_csv(self):
        # Mở hộp thoại để chọn file CSV
        file_path = filedialog.askopenfilename(
            title="Chọn file đầu vào",
            filetypes=[("CSV files", "*.csv")],
            initialdir="D:/IOT CHALLENGE/smart-cart-heatmap/liveheatmap/data/5.map_build_folder/csv"
        )

        # Kiểm tra nếu không chọn file
        if not file_path:
            return

        # Đọc dữ liệu từ file CSV
        try:
            with open(file_path, encoding='utf-8') as f:
                reader = csv.reader(f)
                rows_data = list(reader)
                self.cols = len(rows_data[0]) - 1
                self.rows = len(rows_data) - 1
                self.rect_types = {}

                # Xử lý dữ liệu từ CSV để lưu vào rect_types
                for i in range(1, len(rows_data)):
                    for j in range(1, len(rows_data[i])):
                        val = rows_data[i][j]
                        if val != "Đường đi" and val != "":
                            loai = "Cửa ra vào" if val == "Cửa" else val
                            self.rect_types[(i-1, j-1)] = loai
            self.show_map_ui()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc file: {e}")

    def show_map_ui(self):
        # Xóa canvas và khung nút cũ nếu tồn tại
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.destroy()
        if hasattr(self, 'btn_frame') and self.btn_frame:
            self.btn_frame.destroy()

        # Thiết lập kích thước tối đa cho vùng bản đồ
        MAX_MAP_WIDTH = 1200
        MAX_MAP_HEIGHT = 800
        MARGIN = 40  # Lề cho đánh số hàng/cột

        # Tính kích thước ô lưới tối ưu
        if self.cols > 0 and self.rows > 0:
            cell_size_w = (MAX_MAP_WIDTH - MARGIN) // self.cols
            cell_size_h = (MAX_MAP_HEIGHT - MARGIN) // self.rows
            self.cell_size = min(60, cell_size_w, cell_size_h)
        else:
            self.cell_size = 60

        # Lưu lề và tính kích thước canvas
        self.margin = MARGIN
        canvas_width = self.cols * self.cell_size + self.margin + 180  # 180 cho chú thích
        canvas_height = max(self.rows * self.cell_size + self.margin, 320)

        # Tạo canvas mới để vẽ bản đồ
        self.canvas = tk.Canvas(
            self.master,
            width=canvas_width,
            height=canvas_height,
            bg="white",
            highlightthickness=0
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.selected_cell = None

        # Tạo khung chứa các nút chức năng
        self.btn_frame = tk.Frame(self.master)
        self.btn_frame.pack(pady=5)

        # Vẽ lại bản đồ
        self.draw_grid()

    def draw_grid(self):
        # Xóa nội dung hiện tại trên canvas
        self.canvas.delete("all")

        # Tìm các vùng kết nối và vẽ các ô lưới
        regions = self.find_connected_regions()
        for loai, cells in regions:
            for r, c in cells:
                color = self.type_colors[loai]
                x1 = self.margin + c * self.cell_size
                y1 = self.margin + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

        # Vẽ các đường viền cho các ô lưới
        for i in range(self.rows):
            for j in range(self.cols):
                x1 = self.margin + j * self.cell_size
                y1 = self.margin + i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                loai = self.rect_types.get((i, j), None)
                sides = [True, True, True, True]
                neighbors = [(i-1, j), (i, j+1), (i+1, j), (i, j-1)]
                for idx, (ni, nj) in enumerate(neighbors):
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        if self.rect_types.get((ni, nj), None) == loai and loai is not None:
                            sides[idx] = False
                if loai is None:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.type_colors["Đường đi"], outline="black")
                else:
                    if sides[0]:
                        self.canvas.create_line(x1, y1, x2, y1, fill="black", width=2)
                    if sides[1]:
                        self.canvas.create_line(x2, y1, x2, y2, fill="black", width=2)
                    if sides[2]:
                        self.canvas.create_line(x2, y2, x1, y2, fill="black", width=2)
                    if sides[3]:
                        self.canvas.create_line(x1, y2, x1, y1, fill="black", width=2)

        # Tự động điều chỉnh cỡ chữ cho số hàng/cột
        min_font = 8
        max_font = 16
        font_size = max(min_font, min(max_font, self.cell_size // 2))
        font = ("Arial", font_size, "bold")

        # Vẽ số hàng/cột nếu kích thước ô đủ lớn
        if self.cell_size >= 10:
            # Vẽ số hàng (bên trái)
            for i in range(self.rows):
                y = self.margin + i * self.cell_size + self.cell_size // 2
                self.canvas.create_text(self.margin // 2, y, text=str(i+1), font=font)
            # Vẽ số cột (trên cùng)
            for j in range(self.cols):
                x = self.margin + j * self.cell_size + self.cell_size // 2
                self.canvas.create_text(x, self.margin // 2, text=str(j + 1), font=font)

        # Vẽ chú thích
        self.draw_legend()

    def draw_legend(self):
        # Vẽ chú thích trên canvas
        x0 = self.margin + self.cols * self.cell_size + 30
        y0 = self.margin
        self.canvas.create_text(x0, y0, text="Chú thích:", anchor='w', font=("Arial", 13, "bold"))
        for idx, (loai, color) in enumerate(self.type_colors.items()):
            y = y0 + 30 + idx*32
            self.canvas.create_rectangle(x0, y, x0+28, y+22, fill=color, outline="black")
            self.canvas.create_text(x0+38, y+11, text="Cửa" if loai == "Cửa ra vào" else loai, anchor='w', font=("Arial", 12))

    def on_click(self, event):
        # Xử lý sự kiện nhấp chuột để chọn ô lưới
        col = (event.x - self.margin) // self.cell_size
        row = (event.y - self.margin) // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.selected_cell = (row, col)
            popup_menu = tk.Menu(self.master, tearoff=0)

            # Tạo menu ngữ cảnh với các loại ô
            for loai in self.type_colors:
                if loai != "Đường đi":
                    popup_menu.add_command(
                        label=loai,
                        command=lambda l=loai: self.set_cell_type(l)
                    )
            if (row, col) in self.rect_types:
                popup_menu.add_separator()
                popup_menu.add_command(label="Bỏ chọn ô này", command=self.unset_cell_type)
            try:
                popup_menu.tk_popup(event.x_root, event.y_root)
            finally:
                popup_menu.grab_release()

    def set_cell_type(self, loai):
        # Đặt loại cho ô được chọn
        row, col = self.selected_cell
        self.rect_types[(row, col)] = loai
        self.draw_grid()

    def unset_cell_type(self):
        # Xóa loại của ô được chọn
        row, col = self.selected_cell
        if (row, col) in self.rect_types:
            del self.rect_types[(row, col)]
        self.draw_grid()

    def find_connected_regions(self):
        # Tìm các vùng kết nối có cùng loại
        visited = set()
        regions = []
        for (r, c), loai in self.rect_types.items():
            if (r, c) not in visited:
                region = []
                queue = deque()
                queue.append((r, c))
                visited.add((r, c))
                while queue:
                    cur = queue.popleft()
                    region.append(cur)
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cur[0]+dr, cur[1]+dc
                        if (nr, nc) in self.rect_types and (nr, nc) not in visited and self.rect_types[(nr, nc)] == loai:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                regions.append((loai, region))
        return regions

    def export_image(self):
        # Mở hộp thoại để lưu file PNG
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not file_path:
            return

        # Tạo hình ảnh từ bản đồ
        img_width = self.cols * self.cell_size + self.margin + 180
        img_height = max(self.rows * self.cell_size + self.margin, 320)
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)

        # Tải font chữ
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Vẽ các vùng kết nối
        regions = self.find_connected_regions()
        for loai, cells in regions:
            for r, c in cells:
                color = self.type_colors[loai]
                x1 = self.margin + c * self.cell_size
                y1 = self.margin + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)

        # Vẽ viền và các ô lưới
        for i in range(self.rows):
            for j in range(self.cols):
                x1 = self.margin + j * self.cell_size
                y1 = self.margin + i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                loai = self.rect_types.get((i, j), None)
                sides = [True, True, True, True]
                neighbors = [(i-1, j), (i, j+1), (i+1, j), (i, j-1)]
                for idx, (ni, nj) in enumerate(neighbors):
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        if self.rect_types.get((ni, nj), None) == loai and loai is not None:
                            sides[idx] = False
                if loai is None:
                    draw.rectangle([x1, y1, x2, y2], outline="black", fill=self.type_colors["Đường đi"])
                else:
                    if sides[0]:
                        draw.line([x1, y1, x2, y1], fill="black", width=2)
                    if sides[1]:
                        draw.line([x2, y1, x2, y2], fill="black", width=2)
                    if sides[2]:
                        draw.line([x2, y2, x1, y2], fill="black", width=2)
                    if sides[3]:
                        draw.line([x1, y2, x1, y1], fill="black", width=2)

        # Vẽ số hàng và cột
        for i in range(self.rows):
            y = self.margin + i * self.cell_size + self.cell_size // 2
            draw.text((self.margin // 4, y - 7), str(i+1), fill="black", font=font)
        for j in range(self.cols):
            x = self.margin + j * self.cell_size + self.cell_size // 2
            draw.text((x - 7, self.margin // 4), str(j + 1), fill="black", font=font)

        # Vẽ chú thích
        x0 = self.margin + self.cols * self.cell_size + 30
        y0 = self.margin
        draw.text((x0, y0), "Chú thích:", fill="black", font=font)
        for idx, (loai, color) in enumerate(self.type_colors.items()):
            y = y0 + 30 + idx*32
            draw.rectangle([x0, y, x0+28, y+22], fill=color, outline="black")
            draw.text((x0+38, y+3), "Cửa" if loai == "Cửa ra vào" else loai, fill="black", font=font_small)

        # Lưu hình ảnh
        img.save(file_path)
        messagebox.showinfo("Thành công", f"Đã lưu bản đồ thành {file_path}")

    def export_csv(self):
        # Mở hộp thoại để lưu file CSV
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        # Tạo dữ liệu lưới từ rect_types
        grid_data = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        for (r, c), loai in self.rect_types.items():
            grid_data[r][c] = "Cửa" if loai == "Cửa ra vào" else loai
        for i in range(self.rows):
            for j in range(self.cols):
                if not grid_data[i][j]:
                    grid_data[i][j] = "Đường đi"

        # Lưu dữ liệu vào file CSV
        try:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                header = [""] + [str(i+1) for i in range(self.cols)]
                writer.writerow(header)
                for i in range(self.rows):
                    row_label = str(i+1)
                    writer.writerow([row_label] + grid_data[i])
            messagebox.showinfo("Thành công", f"Đã lưu bản đồ thành {file_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu file CSV: {e}")

    def get_map_image(self):
        # Tạo hình ảnh từ bản đồ hiện tại
        img_width = self.cols * self.cell_size + self.margin
        img_height = self.rows * self.cell_size + self.margin
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)

        # Tải font chữ
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        # Vẽ các vùng kết nối
        regions = self.find_connected_regions()
        for loai, cells in regions:
            for r, c in cells:
                color = self.type_colors[loai]
                x1 = self.margin + c * self.cell_size
                y1 = self.margin + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)

        # Vẽ viền và các ô lưới
        for i in range(self.rows):
            for j in range(self.cols):
                x1 = self.margin + j * self.cell_size
                y1 = self.margin + i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                loai = self.rect_types.get((i, j), None)
                sides = [True, True, True, True]
                neighbors = [(i-1, j), (i, j+1), (i+1, j), (i, j-1)]
                for idx, (ni, nj) in enumerate(neighbors):
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        if self.rect_types.get((ni, nj), None) == loai and loai is not None:
                            sides[idx] = False
                if loai is None:
                    draw.rectangle([x1, y1, x2, y2], outline="black", fill=self.type_colors["Đường đi"])
                else:
                    if sides[0]:
                        draw.line([x1, y1, x2, y1], fill="black", width=2)
                    if sides[1]:
                        draw.line([x2, y1, x2, y2], fill="black", width=2)
                    if sides[2]:
                        draw.line([x2, y2, x1, y2], fill="black", width=2)
                    if sides[3]:
                        draw.line([x1, y2, x1, y1], fill="black", width=2)

        # Vẽ số hàng và cột
        for i in range(self.rows):
            y = self.margin + i * self.cell_size + self.cell_size // 2
            draw.text((self.margin // 4, y - 7), str(i+1), fill="black", font=font)
        for j in range(self.cols):
            x = self.margin + j * self.cell_size + self.cell_size // 2
            draw.text((x - 7, self.margin // 4), str(j + 1), fill="black", font=font)
        return img

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

if __name__ == "__main__":
    # Khởi chạy ứng dụng
    root = tk.Tk()
    app = GridMapApp(root)
    root.mainloop()
