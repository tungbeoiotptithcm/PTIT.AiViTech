import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import csv

def get_display_text(loai):
    return "Cửa" if loai == "Cửa ra vào" else loai

class GridMapApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Grid Map")

        self.rows = simpledialog.askinteger("Số hàng", "Nhập số hàng:", minvalue=1, maxvalue=26)
        self.cols = simpledialog.askinteger("Số cột", "Nhập số cột:", minvalue=1, maxvalue=100)
        if not self.rows or not self.cols:
            self.master.destroy()
            return

        self.cell_size = 60
        self.margin = 40
        # Tăng chiều rộng canvas để đủ chỗ vẽ legend bên phải
        canvas_width = self.cols * self.cell_size + self.margin + 180
        canvas_height = max(self.rows * self.cell_size + self.margin, 320)

        self.canvas = tk.Canvas(master, width=canvas_width, height=canvas_height, bg="white", highlightthickness=0)
        self.canvas.pack()
        self.rect_types = {}  # (row, col): loại đã chọn

        self.type_colors = {
            "Cửa ra vào": "red",
            "Đồ uống": "lightblue",
            "Gia vị": "orange",
            "Nhu yếu phẩm": "lightgreen",
            "Lương thực": "khaki",
            "Bánh kẹo": "pink",
            "Thu ngân": "brown",
            "Đường đi": "#eeeeee"   # Màu xám nhạt
        }

        self.draw_grid()
        self.canvas.bind("<Button-1>", self.on_click)
        self.selected_cell = None

        btn_save = tk.Button(master, text="Xuất file PNG", command=self.export_image)
        btn_save.pack(pady=5)
        btn_csv = tk.Button(master, text="Xuất file CSV", command=self.export_csv)
        btn_csv.pack(pady=5)

    def draw_grid(self):
        self.canvas.delete("all")
        regions = self.find_connected_regions()
        # Vẽ màu nền cho các vùng liên thông (không vẽ text trong ô)
        for loai, cells in regions:
            for r, c in cells:
                color = self.type_colors[loai]
                x1 = self.margin + c * self.cell_size
                y1 = self.margin + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

        # Vẽ viền ngoài cho từng ô, chỉ vẽ cạnh ngoài vùng
        for i in range(self.rows):
            for j in range(self.cols):
                x1 = self.margin + j * self.cell_size
                y1 = self.margin + i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                loai = self.rect_types.get((i, j), None)
                sides = [True, True, True, True]  # top, right, bottom, left
                neighbors = [ (i-1, j), (i, j+1), (i+1, j), (i, j-1) ]
                for idx, (ni, nj) in enumerate(neighbors):
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        if self.rect_types.get((ni, nj), None) == loai and loai is not None:
                            sides[idx] = False
                if loai is None:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.type_colors["Đường đi"], outline="black")
                else:
                    if sides[0]:  # top
                        self.canvas.create_line(x1, y1, x2, y1, fill="black", width=2)
                    if sides[1]:  # right
                        self.canvas.create_line(x2, y1, x2, y2, fill="black", width=2)
                    if sides[2]:  # bottom
                        self.canvas.create_line(x2, y2, x1, y2, fill="black", width=2)
                    if sides[3]:  # left
                        self.canvas.create_line(x1, y2, x1, y1, fill="black", width=2)

        # Vẽ nhãn hàng (A-Z)
        for i in range(self.rows):
            y = self.margin + i * self.cell_size + self.cell_size // 2
            self.canvas.create_text(self.margin // 2, y, text=chr(65 + i), font=("Arial", 14, "bold"))
        # Vẽ nhãn cột (1, 2, ...)
        for j in range(self.cols):
            x = self.margin + j * self.cell_size + self.cell_size // 2
            self.canvas.create_text(x, self.margin // 2, text=str(j + 1), font=("Arial", 14, "bold"))

        # Vẽ chú thích bên phải (legend)
        self.draw_legend()

    def draw_legend(self):
        # Vẽ legend bên phải lưới
        x0 = self.margin + self.cols * self.cell_size + 30
        y0 = self.margin
        self.canvas.create_text(x0, y0, text="Chú thích:", anchor='w', font=("Arial", 13, "bold"))
        for idx, (loai, color) in enumerate(self.type_colors.items()):
            y = y0 + 30 + idx*32
            self.canvas.create_rectangle(x0, y, x0+28, y+22, fill=color, outline="black")
            self.canvas.create_text(x0+38, y+11, text=get_display_text(loai), anchor='w', font=("Arial", 12))

    def on_click(self, event):
        col = (event.x - self.margin) // self.cell_size
        row = (event.y - self.margin) // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.selected_cell = (row, col)
            # Tạo popup menu động
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
            popup_menu.tk_popup(event.x_root, event.y_root)

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
                neighbors = [ (i-1, j), (i, j+1), (i+1, j), (i, j-1) ]
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
        # Vẽ nhãn hàng
        for i in range(self.rows):
            y = self.margin + i * self.cell_size + self.cell_size // 2
            draw.text((self.margin // 4, y - 7), chr(65 + i), fill="black", font=font)
        # Vẽ nhãn cột
        for j in range(self.cols):
            x = self.margin + j * self.cell_size + self.cell_size // 2
            draw.text((x - 7, self.margin // 4), str(j + 1), fill="black", font=font)
        # Vẽ legend bên phải
        x0 = self.margin + self.cols * self.cell_size + 30
        y0 = self.margin
        draw.text((x0, y0), "Chú thích:", fill="black", font=font)
        for idx, (loai, color) in enumerate(self.type_colors.items()):
            y = y0 + 30 + idx*32
            draw.rectangle([x0, y, x0+28, y+22], fill=color, outline="black")
            draw.text((x0+38, y+3), get_display_text(loai), fill="black", font=font_small)
        img.save(file_path)
        messagebox.showinfo("Thành công", f"Đã lưu bản đồ thành {file_path}")

    def export_csv(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        grid_data = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        for (r, c), loai in self.rect_types.items():
            grid_data[r][c] = get_display_text(loai)
        # Gán "Đường đi" cho ô chưa chọn loại
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
                    row_label = chr(65 + i)
                    writer.writerow([row_label] + grid_data[i])
            messagebox.showinfo("Thành công", f"Đã lưu bản đồ thành {file_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu file CSV: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GridMapApp(root)
    root.mainloop()
