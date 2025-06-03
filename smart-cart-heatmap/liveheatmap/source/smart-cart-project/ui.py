import tkinter as tk
from tkinter import filedialog, messagebox
#from map_handler import GridMapApp  # Import lớp GridMapApp từ map_handler
import csv
from PIL import Image, ImageDraw, ImageFont


class GridMapApp:
    def __init__(self, master):  # Đã sửa: Đảm bảo nhận tham số master
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
            initialdir="D:/IOT CHALLENGE/smart-cart-heatmap/liveheatmap/data/map_build_folder/csv"
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

    def export_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not file_path:
            return

        # Kích thước hình ảnh xuất ra
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

        # Vẽ các vùng đã chọn màu theo loại
        for (r, c), loai in self.rect_types.items():
            color = self.type_colors.get(loai, "white")
            x1 = self.margin + c * self.cell_size
            y1 = self.margin + r * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            draw.rectangle([x1, y1, x2, y2], fill=color, outline="black")

        # Vẽ số thứ tự hàng
        for i in range(self.rows):
            y = self.margin + i * self.cell_size + self.cell_size // 2
            draw.text((self.margin // 4, y - 7), str(i + 1), fill="black", font=font)

        # Vẽ số thứ tự cột
        for j in range(self.cols):
            x = self.margin + j * self.cell_size + self.cell_size // 2
            draw.text((x - 7, self.margin // 4), str(j + 1), fill="black", font=font)

        # Vẽ bảng chú thích bên phải
        x0 = self.margin + self.cols * self.cell_size + 30
        y0 = self.margin
        draw.text((x0, y0), "Chú thích:", fill="black", font=font)
        for idx, (loai, color) in enumerate(self.type_colors.items()):
            y = y0 + 30 + idx * 32
            draw.rectangle([x0, y, x0 + 28, y + 22], fill=color, outline="black")
            draw.text((x0 + 38, y + 3), "Cửa" if loai == "Cửa ra vào" else loai, fill="black", font=font_small)

        # Lưu ảnh và hiển thị thông báo thành công
        img.save(file_path)
        messagebox.showinfo("Thành công", f"Đã lưu bản đồ thành {file_path}")


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

        