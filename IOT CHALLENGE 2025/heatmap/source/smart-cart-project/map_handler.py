import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import csv

class GridMapApp:
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