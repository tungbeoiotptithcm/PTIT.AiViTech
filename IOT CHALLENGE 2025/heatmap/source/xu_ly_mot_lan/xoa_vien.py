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
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
import threading
import pandas as pd
from scipy.stats import percentileofscore


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
        #self.input_frame.pack(pady=10)

        self.input_frame.pack(pady=(30, 30))  # padding trên-dưới lớn hơn

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
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
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
        if self.canvas:
            self.canvas.destroy()
        if self.btn_frame:
            self.btn_frame.destroy()

        self.cell_size = 60
        self.margin = 0
        canvas_width = self.cols * self.cell_size + self.margin + 180
        canvas_height = max(self.rows * self.cell_size + self.margin, 320)
        self.canvas = tk.Canvas(self.master, width=canvas_width, height=canvas_height, bg="white", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.selected_cell = None

        self.btn_frame = tk.Frame(self.master)
        self.btn_frame.pack(pady=5)
        #
        spacer = tk.Label(self.master, text="", height=1) 
        spacer.pack()
        #
        btn_save = tk.Button(self.btn_frame, text="Xuất file PNG", command=self.export_image)
        btn_save.pack(side=tk.LEFT, padx=5)
        btn_csv = tk.Button(self.btn_frame, text="Xuất file CSV", command=self.export_csv)
        btn_csv.pack(side=tk.LEFT, padx=5)
        btn_heatmap = tk.Button(self.btn_frame, text="Tạo Heatmap", command=self.create_heatmap)
        btn_heatmap.pack(side=tk.LEFT, padx=5)

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
        img_width = self.cols * self.cell_size + 180
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
        img_width = self.cols * self.cell_size
        img_height = self.rows * self.cell_size
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
        
        
        return img


    def create_heatmap(self):
        if self.rows is None or self.cols is None:
            messagebox.showerror("Lỗi", "Vui lòng tạo bản đồ trước khi tạo heatmap.")
            return
    
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not video_path:
            messagebox.showinfo("Thông báo", "Không có file video nào được chọn.")
            return
    
        def worker():
            try:
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
                                cx = (x1 + x2) // 2
                                cy = (y1 + y2) // 2
                                heatmap_data.append((cx, cy))
                                frame_coordinates.append((cx, cy))
                    all_coordinates.append(frame_coordinates)
    
                cap.release()
                cv2.destroyAllWindows()
    
                if len(heatmap_data) > 0:
                    points = np.array(heatmap_data).T
                    xgrid = np.linspace(0, 640, 640)
                    ygrid = np.linspace(0, 480, 480)
                    X, Y = np.meshgrid(xgrid, ygrid)
                    kde = gaussian_kde(points, bw_method=0.2)
                    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
                    # Map nền và heatmap nền
                    map_img = self.get_map_image().resize((640, 480))
                    map_array = np.array(map_img)
    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(map_array, alpha=0.9)
                    plt.imshow(Z, extent=[0, 640, 480, 0], cmap='jet', alpha=0.4)
                    contour = plt.contour(X, Y, Z, levels=10, colors='red', linewidths=1.5)
                    plt.clabel(contour, inline=1, fontsize=8, fmt="%.2f")
    
                    plt.title("Heatmap & Contour mật độ người trên bản đồ siêu thị", fontsize=14)
                    plt.axis('off')
                    plt.tight_layout()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"heatmap_contour_{timestamp}.png"
                    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1, dpi=150)
                    plt.close()
                    self.master.after(0, lambda: messagebox.showinfo("Thành công", f"Đã lưu heatmap tại: {fname}"))
    
                    # PHÂN TÍCH VÀ TẠO GỢI Ý
                    region_names = {
                        (100, 100): "Khu đồ uống",
                        (300, 100): "Khu gia vị",
                        (500, 100): "Khu lương thực",
                        (100, 300): "Khu thu ngân",
                        (300, 300): "Lối đi chính",
                        (500, 300): "Khu bánh kẹo"
                    }
    
                    region_info = []
                    all_densities = Z.ravel()
    
                    for (x, y), name in region_names.items():
                        if 0 <= x < 640 and 0 <= y < 480:
                            # Lấy giá trị trung bình vùng 11x11 quanh tâm thay vì chỉ 1 điểm
                            density = np.mean(Z[max(0, y-5):min(480, y+6), max(0, x-5):min(640, x+6)])
                            percentile = percentileofscore(all_densities, density)
    
                            suggestion = ""
                            if percentile >= 75:
                                suggestion = "Mở rộng khu vực hoặc thêm quầy phục vụ"
                            elif percentile >= 50:
                                suggestion = "Tối ưu bố trí sản phẩm"
                            elif percentile >= 25:
                                suggestion = "Duy trì hiện trạng"
                            else:
                                suggestion = "Cân nhắc giảm diện tích"
    
                            region_info.append({
                                'Khu vực': name,
                                'Tọa độ X': x,
                                'Tọa độ Y': y,
                                'Mật độ': density,
                                'Phần trăm': f"{percentile:.1f}%",
                                'Đề xuất': suggestion
                            })
    
                    # Tạo DataFrame
                    df = pd.DataFrame(region_info)
    
                    # Tạo ảnh gợi ý
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(df['Tọa độ X'], df['Tọa độ Y'], c=df['Mật độ'], cmap='coolwarm', s=200, edgecolors='black')
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Mật độ')
    
                    for i, row in df.iterrows():
                        ax.text(row['Tọa độ X'], row['Tọa độ Y'] + 20, f"{row['Khu vực']}", ha='center', fontsize=10, fontweight='bold')
                        ax.text(row['Tọa độ X'], row['Tọa độ Y'] - 20, f"{row['Đề xuất']}", ha='center', fontsize=8, color='gray')
    
                    ax.set_title('Gợi ý chỉnh sửa cửa hàng theo khu vực')
                    ax.set_xlim(0, 640)
                    ax.set_ylim(0, 480)
                    ax.invert_yaxis()
                    ax.axis('off')
    
                    plt.tight_layout()
                    suggestion_img_filename = f"store_suggestion_map_{timestamp}.png"
                    fig.savefig(suggestion_img_filename)
                    plt.close(fig)
    
                    # Xuất file CSV gợi ý
                    csv_suggestion_filename = f"store_optimization_{timestamp}.csv"
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
                    csv_filename = f"person_coordinates_{timestamp}.csv"
                    df_coords.to_csv(csv_filename, index=False)
                    self.master.after(0, lambda: messagebox.showinfo("Thành công", f"Tọa độ đã được lưu vào file: {csv_filename}"))
    
            except Exception as e:
                self.master.after(0, lambda: messagebox.showerror("Lỗi", f"Không thể tạo heatmap: {str(e)}"))
    
        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = GridMapApp(root)
    root.mainloop()
