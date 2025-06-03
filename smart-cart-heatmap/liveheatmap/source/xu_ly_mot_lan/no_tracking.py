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
        self.master = master
        self.master.title("Grid Map")
        self.rows = None
        self.cols = None
        self.rect_types = {}

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

        # Khung nhập số hàng/cột và chọn mở file
        self.input_frame = tk.Frame(master)
        self.input_frame.pack(pady=10)

        tk.Label(self.input_frame, text="Số hàng (2-99):").grid(row=0, column=0, padx=5)
        self.entry_rows = tk.Entry(self.input_frame, width=7)
        self.entry_rows.grid(row=0, column=1, padx=5)
        tk.Label(self.input_frame, text="Số cột (2-99):").grid(row=0, column=2, padx=5)
        self.entry_cols = tk.Entry(self.input_frame, width=7)
        self.entry_cols.grid(row=0, column=3, padx=5)

        self.btn_create = tk.Button(self.input_frame, text="Tạo map mới", command=self.create_new_map)
        self.btn_create.grid(row=0, column=4, padx=8)
        self.btn_open = tk.Button(self.input_frame, text="Mở file CSV", command=self.open_csv)
        self.btn_open.grid(row=0, column=5, padx=8)
        
        # Thêm các nút chức năng lên cùng hàng
        self.btn_export_png = tk.Button(self.input_frame, text="Xuất file PNG", command=self.export_image)
        self.btn_export_png.grid(row=0, column=6, padx=5)
        self.btn_export_csv = tk.Button(self.input_frame, text="Xuất file CSV", command=self.export_csv)
        self.btn_export_csv.grid(row=0, column=7, padx=5)
        self.btn_heatmap = tk.Button(self.input_frame, text="Tạo Heatmap", command=self.create_heatmap)
        self.btn_heatmap.grid(row=0, column=8, padx=5)

        self.canvas = None
        self.btn_frame = None

    def create_new_map(self):
        try:
            rows = int(self.entry_rows.get())
            cols = int(self.entry_cols.get())
            if rows < 2 or rows > 99 or cols < 2 or cols > 99:
                raise ValueError
        except:
            messagebox.showerror("Lỗi", "Số hàng và số cột phải là số nguyên từ 2 đến 99.")
            return
        self.rows = rows
        self.cols = cols
        self.rect_types = {}
        self.show_map_ui()

    def open_csv(self):

        file_path = filedialog.askopenfilename(
            title="Chọn CSV đầu vào",
            filetypes=[("CSV files", "*.csv")],
            initialdir="D:/IOT CHALLENGE/smart-cart-heatmap/liveheatmap/data/map_build_folder/csv"
        )
        
        if not file_path:
            return
        try:
            with open(file_path, encoding='utf-8') as f:
                reader = csv.reader(f)
                rows_data = list(reader)
                self.cols = len(rows_data[0]) - 1
                self.rows = len(rows_data) - 1
                self.rect_types = {}
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
        # Xóa canvas và frame cũ nếu có
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.destroy()
        if hasattr(self, 'btn_frame') and self.btn_frame:
            self.btn_frame.destroy()

        # Thông số tối đa cho vùng bản đồ (không tính chú thích)
        MAX_MAP_WIDTH = 1200
        MAX_MAP_HEIGHT = 800
        MARGIN = 40  # Lề cho đánh số hàng/cột

        # Tính kích thước ô tối ưu
        if self.cols > 0 and self.rows > 0:
            cell_size_w = (MAX_MAP_WIDTH - MARGIN) // self.cols
            cell_size_h = (MAX_MAP_HEIGHT - MARGIN) // self.rows
            self.cell_size = min(60, cell_size_w, cell_size_h)
        else:
            self.cell_size = 60

        self.margin = MARGIN
        canvas_width = self.cols * self.cell_size + self.margin + 180  # 180 cho chú thích bên phải
        canvas_height = max(self.rows * self.cell_size + self.margin, 320)

        # Tạo canvas mới
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

        # Tạo frame chứa các nút chức năng
        self.btn_frame = tk.Frame(self.master)
        self.btn_frame.pack(pady=5)

        # tk.Button(self.btn_frame, text="Xuất file PNG", command=self.export_image).pack(side=tk.LEFT, padx=5)
        # tk.Button(self.btn_frame, text="Xuất file CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)
        # tk.Button(self.btn_frame, text="Tạo Heatmap", command=self.create_heatmap).pack(side=tk.LEFT, padx=5)

        # Vẽ lại bản đồ
        self.draw_grid()


    def draw_grid(self):
        self.canvas.delete("all")
        regions = self.find_connected_regions()
        for loai, cells in regions:
            for r, c in cells:
                color = self.type_colors[loai]
                x1 = self.margin + c * self.cell_size
                y1 = self.margin + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

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

        # --- Tự động điều chỉnh cỡ font cho số hàng/cột ---
        min_font = 8
        max_font = 16
        font_size = max(min_font, min(max_font, self.cell_size // 2))
        font = ("Arial", font_size, "bold")

        # Nếu cell_size quá nhỏ, không vẽ số để tránh rối
        if self.cell_size >= 10:
            # Vẽ số hàng (bên trái)
            for i in range(self.rows):
                y = self.margin + i * self.cell_size + self.cell_size // 2
                self.canvas.create_text(self.margin // 2, y, text=str(i+1), font=font)
            # Vẽ số cột (trên cùng)
            for j in range(self.cols):
                x = self.margin + j * self.cell_size + self.cell_size // 2
                self.canvas.create_text(x, self.margin // 2, text=str(j + 1), font=font)

        self.draw_legend()


    def draw_legend(self):
        x0 = self.margin + self.cols * self.cell_size + 30
        y0 = self.margin
        self.canvas.create_text(x0, y0, text="Chú thích:", anchor='w', font=("Arial", 13, "bold"))
        for idx, (loai, color) in enumerate(self.type_colors.items()):
            y = y0 + 30 + idx*32
            self.canvas.create_rectangle(x0, y, x0+28, y+22, fill=color, outline="black")
            self.canvas.create_text(x0+38, y+11, text="Cửa" if loai == "Cửa ra vào" else loai, anchor='w', font=("Arial", 12))

    def on_click(self, event):
        col = (event.x - self.margin) // self.cell_size
        row = (event.y - self.margin) // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.selected_cell = (row, col)
            popup_menu = tk.Menu(self.master, tearoff=0)
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
        row, col = self.selected_cell
        self.rect_types[(row, col)] = loai
        self.draw_grid()

    def unset_cell_type(self):
        row, col = self.selected_cell
        if (row, col) in self.rect_types:
            del self.rect_types[(row, col)]
        self.draw_grid()

    def find_connected_regions(self):
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
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not file_path:
            return
        img_width = self.cols * self.cell_size + self.margin + 180
        img_height = max(self.rows * self.cell_size + self.margin, 320)
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
        regions = self.find_connected_regions()
        for loai, cells in regions:
            for r, c in cells:
                color = self.type_colors[loai]
                x1 = self.margin + c * self.cell_size
                y1 = self.margin + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
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
        for i in range(self.rows):
            y = self.margin + i * self.cell_size + self.cell_size // 2
            draw.text((self.margin // 4, y - 7), str(i+1), fill="black", font=font)
        for j in range(self.cols):
            x = self.margin + j * self.cell_size + self.cell_size // 2
            draw.text((x - 7, self.margin // 4), str(j + 1), fill="black", font=font)
        x0 = self.margin + self.cols * self.cell_size + 30
        y0 = self.margin
        draw.text((x0, y0), "Chú thích:", fill="black", font=font)
        for idx, (loai, color) in enumerate(self.type_colors.items()):
            y = y0 + 30 + idx*32
            draw.rectangle([x0, y, x0+28, y+22], fill=color, outline="black")
            draw.text((x0+38, y+3), "Cửa" if loai == "Cửa ra vào" else loai, fill="black", font=font_small)
        img.save(file_path)
        messagebox.showinfo("Thành công", f"Đã lưu bản đồ thành {file_path}")

    def export_csv(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        grid_data = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        for (r, c), loai in self.rect_types.items():
            grid_data[r][c] = "Cửa" if loai == "Cửa ra vào" else loai
        for i in range(self.rows):
            for j in range(self.cols):
                if not grid_data[i][j]:
                    grid_data[i][j] = "Đường đi"
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
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        regions = self.find_connected_regions()
        for loai, cells in regions:
            for r, c in cells:
                color = self.type_colors[loai]
                x1 = self.margin + c * self.cell_size
                y1 = self.margin + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
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
        for i in range(self.rows):
            y = self.margin + i * self.cell_size + self.cell_size // 2
            draw.text((self.margin // 4, y - 7), str(i+1), fill="black", font=font)
        for j in range(self.cols):
            x = self.margin + j * self.cell_size + self.cell_size // 2
            draw.text((x - 7, self.margin // 4), str(j + 1), fill="black", font=font)
        return img


    def create_heatmap(self):
        if self.rows is None or self.cols is None:
            messagebox.showerror("Lỗi", "Vui lòng tạo bản đồ trước khi tạo heatmap.")
            return

        video_path = filedialog.askopenfilename(
            title="Chọn video đầu vào",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")],
            initialdir="D:/IOT CHALLENGE/smart-cart-heatmap/liveheatmap/data/video"  # hoặc đường dẫn tuyệt đối tùy bạn
        )

        
        if not video_path:
            messagebox.showinfo("Thông báo", "Không có file video nào được chọn.")
            return
    
        def worker():
            try:
                margin = getattr(self, 'margin', 40)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    messagebox.showerror("Lỗi", "Không thể mở file video.")
                    return
    
                model = YOLO("yolov8n.pt")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model.to(device)
    
                heatmap_data = []
                all_coordinates = []
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_skip = int(fps) if fps > 0 else 30
                frame_count = 0
    
                self.master.after(0, lambda: messagebox.showinfo("Đang xử lý", "Đang tạo heatmap, vui lòng chờ..."))
    
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue
                    
                    frame = cv2.resize(frame, (640, 480))
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model.predict(rgb_frame, verbose=False)
    
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
    
                cap.release()
                cv2.destroyAllWindows()
    
                if len(heatmap_data) > 0:
                    points = np.array(heatmap_data).T
                    xgrid = np.linspace(0, 640 - 2*margin, 640 - 2*margin)
                    ygrid = np.linspace(0, 480 - 2*margin, 480 - 2*margin)
                    X, Y = np.meshgrid(xgrid, ygrid)
                    kde = gaussian_kde(points, bw_method=0.2)
                    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
                    # Map nền và heatmap nền
                    map_img = self.get_map_image().resize((640, 480))
                    map_array = np.array(map_img)
                    margin = self.margin
                    map_img_full = self.get_map_image()
                    width, height = map_img_full.size
                    cropped_img = map_img_full.crop((margin, margin, width, height)) 
                    map_img = cropped_img.resize((640 - 2*margin, 480 - 2*margin))
    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(map_img, alpha=0.9, extent=[margin, 640 - margin, 480 - margin, margin])  # Điều chỉnh vị trí
                    plt.imshow(Z, extent=[margin, 640 - margin, 480 - margin, margin], cmap='jet', alpha=0.4)
    
                    plt.title("Heatmap & Contour mật độ người trên bản đồ siêu thị", fontsize=14)
                    plt.axis('off')
                    plt.tight_layout()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"D:\IOT CHALLENGE\smart-cart-heatmap\liveheatmap\data\heatmap_contour_folder\heatmap_contour_{timestamp}.png"
                    plt.savefig(fname, bbox_inches='tight', dpi=150)
                    plt.close()
                    self.master.after(0, lambda: messagebox.showinfo("Thành công", f"Đã lưu heatmap tại: {fname}"))
    
                    # PHÂN TÍCH VÀ TẠO GỢI Ý
                    # --- Tạo region_names chuẩn theo scale ảnh heatmap ---
                    region_info = []
                    regions = self.find_connected_regions()
                    map_img_full = self.get_map_image()
                    width, height = map_img_full.size

                    # Resize chuẩn như dùng để tạo heatmap
                    scaled_width = 640 - 2 * margin
                    scaled_height = 480 - 2 * margin
                    scale_x = scaled_width / (width - margin)
                    scale_y = scaled_height / (height - margin)

                    all_densities = Z.ravel()

                    for loai, cells in regions:
                        if loai != "Đường đi":
                            xs = [c * self.cell_size + self.margin + self.cell_size // 2 for _, c in cells]
                            ys = [r * self.cell_size + self.margin + self.cell_size // 2 for r, _ in cells]
                            cx = int(sum(xs) / len(xs))
                            cy = int(sum(ys) / len(ys))

                            #Scale về tọa độ ảnh heatmap
                            x_scaled = int((cx - margin) * scale_x)
                            y_scaled = int((cy - margin) * scale_y)

                            # Đảm bảo không vượt giới hạn heatmap
                            if 0 <= x_scaled < Z.shape[1] and 0 <= y_scaled < Z.shape[0]:
                                # Tính mật độ theo vùng nhỏ quanh điểm
                                density = np.mean(Z[max(0, y_scaled - 5):min(Z.shape[0], y_scaled + 6),
                                                    max(0, x_scaled - 5):min(Z.shape[1], x_scaled + 6)])
                                percentile = percentileofscore(all_densities, density)

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

    
                    # Tạo DataFrame
                    df = pd.DataFrame(region_info)
    
                    # Tạo ảnh gợi ý
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Đặt map_img (đã resize) làm nền
                    ax.imshow(map_img, extent=[0, Z.shape[1], Z.shape[0], 0], alpha=1.0)

                    # Vẽ chấm gợi ý đúng vị trí
                    scatter = ax.scatter(
                        df['Tọa độ X'],
                        df['Tọa độ Y'],
                        c=df['Mật độ'],
                        cmap='coolwarm',
                        s=200,
                        edgecolors='black'
                    )

                    # Colorbar mật độ
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Mật độ')

                    # Hiển thị nhãn và gợi ý
                    for i, row in df.iterrows():
                        ax.text(row['Tọa độ X'], row['Tọa độ Y'] - 10, f"{row['Khu vực']}", ha='center',
                                fontsize=10, fontweight='bold', color='black')
                        ax.text(row['Tọa độ X'], row['Tọa độ Y'] + 20, f"{row['Đề xuất']}", ha='center',
                                fontsize=8, color='gray')

                    ax.set_title('Store Suggestion Map')
                    ax.set_xlim(0, Z.shape[1])
                    ax.set_ylim(Z.shape[0], 0)  # vì trục y là ảnh
                    ax.axis('off')

                    plt.tight_layout()
                    suggestion_img_filename = f"D:\IOT CHALLENGE\smart-cart-heatmap\liveheatmap\data\store_suggestion_map\store_suggestion_map_{timestamp}.png"
                    plt.savefig(suggestion_img_filename, dpi=150)
                    plt.close(fig)

    
                    # Xuất file CSV gợi ý
                    csv_suggestion_filename = f"D:\IOT CHALLENGE\smart-cart-heatmap\liveheatmap\data\store_optimization_folder\store_optimization_{timestamp}.csv"
                    df.to_csv(csv_suggestion_filename, index=False)
    
                    self.master.after(0, lambda: messagebox.showinfo("Gợi ý tối ưu", 
                        f"Đã lưu gợi ý tại: {suggestion_img_filename}\n\n" +
                        "Các đề xuất chính:\n" + 
                        "\n".join([f"- {row['Khu vực']}: {row['Đề xuất']}" for i, row in df.iterrows()])
                    ))
    
                else:
                    self.master.after(0, lambda: messagebox.showwarning("Không có dữ liệu", "Không có dữ liệu heatmap để vẽ contour."))
    
                # Xuất tọa độ ra file CSV
                if all_coordinates:
                    flat_coords = []
                    for frame_number, frame_coords in enumerate(all_coordinates):
                        for person_number, (x, y) in enumerate(frame_coords):
                            flat_coords.append([frame_number + 1, person_number + 1, x, y])
    
                    df_coords = pd.DataFrame(flat_coords, columns=['Frame', 'Person', 'X', 'Y'])
                    csv_filename = f"D:\IOT CHALLENGE\smart-cart-heatmap\liveheatmap\data\person_coordinates_folder\person_coordinates_{timestamp}.csv"
                    df_coords.to_csv(csv_filename, index=False)
                    self.master.after(0, lambda: messagebox.showinfo("Thành công", f"Tọa độ đã được lưu vào file: {csv_filename}"))
    
            except Exception as e:
                self.master.after(0, lambda e=e: messagebox.showerror("Lỗi", f"Không thể tạo heatmap: {str(e)}"))
    
        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = GridMapApp(root)
    root.mainloop()

#xuất file theo folder - không có live tracking