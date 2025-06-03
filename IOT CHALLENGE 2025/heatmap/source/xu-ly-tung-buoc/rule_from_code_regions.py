import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import re
import matplotlib.pyplot as plt

# === Bước 1: Trích xuất region_names từ file heatmap gốc ===
def extract_region_names_from_py(py_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        code = f.read()
    match = re.search(r'region_names\s*=\s*\{(.*?)\}', code, re.DOTALL)
    region_str = match.group(1)
    region_lines = region_str.split('\n')
    region_dict = {}
    for line in region_lines:
        m = re.match(r'\s*\((\d+),\s*(\d+)\):\s*"([^"]+)"', line.strip())
        if m:
            x, y, name = int(m[1]), int(m[2]), m[3]
            region_dict[name] = (x, y)
    return region_dict

# === Cập nhật đường dẫn file code gốc ===
region_names = extract_region_names_from_py("D:/IOT CHALLENGE/smart-cart-heatmap/liveheatmap/source/xu-ly-tung-buoc/contour_csv.py")

# === Bước 2: Đọc tọa độ người từ file CSV ===
df = pd.read_csv("person_coordinates_20250515_152253.csv")
coordinates = df[['X', 'Y']].values.T

# === Bước 3: Tính mật độ KDE ===
kde = gaussian_kde(coordinates)
x_grid = np.linspace(0, 640, 640)
y_grid = np.linspace(0, 480, 480)
X, Y = np.meshgrid(x_grid, y_grid)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# === Bước 4: Tính toán mật độ và lưu trữ thông tin cho trực quan hóa ===
region_info = []
for name, (x, y) in region_names.items():
    if 0 <= x < 640 and 0 <= y < 480:
        density = Z[int(y), int(x)]
        percentile_rank = (np.sum(Z <= density) / Z.size) * 100
        suggestion = ""
        color = "white"
        if percentile_rank >= 80:
            suggestion = "Cần tăng cường"
            color = "red"
        elif percentile_rank >= 60:
            suggestion = "Nên bổ sung"
            color = "orange"
        elif percentile_rank >= 40:
            suggestion = "Theo dõi"
            color = "yellow"
        elif percentile_rank >= 20:
            suggestion = "Điều phối"
            color = "blue"
        else:
            suggestion = "Cân nhắc giảm"
            color = "green"
        region_info.append({'name': name, 'density': density, 'percentile': percentile_rank, 'suggestion': suggestion, 'color': color})

# === Bước 5: Sắp xếp dữ liệu theo hạng phần trăm từ cao xuống thấp ===
region_info_sorted = sorted(region_info, key=lambda item: item['percentile'], reverse=True)

# === Bước 6: Tạo hình ảnh kết hợp biểu đồ cột và text ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))  # 1 hàng, 2 cột

# --- Biểu đồ cột (nửa bên trái) ---
bar_names = [info['name'] for info in region_info_sorted]
percentiles = [info['percentile'] for info in region_info_sorted]
colors = [info['color'] for info in region_info_sorted]

ax1.bar(bar_names, percentiles, color=colors)
ax1.set_xlabel('Khu vực', fontsize=12)
ax1.set_ylabel('Hạng phần trăm mật độ', fontsize=12)
ax1.set_title('Mật độ người theo khu vực (Sắp xếp từ cao xuống thấp)', fontsize=14)
ax1.tick_params(axis='y', labelsize=10)
ax1.set_xticklabels(bar_names, rotation=45, ha="right", fontsize=10) # Thiết lập labels và rotation riêng

# --- Text gợi ý (nửa bên phải) ---
text_output = "=== GỢI Ý PHÂN BỐ NHÂN SỰ ===\n"
for info in region_info_sorted:
    text_output += f"- Khu vực {info['name']} ({info['percentile']:.2f}%): {info['suggestion']}\n"

ax2.text(0.05, 0.95, text_output, transform=ax2.transAxes, fontsize=15, verticalalignment='top')
ax2.axis('off')  # Tắt trục của nửa bên phải

plt.tight_layout()

# Lưu hình ảnh
output_filename = f"bieu_do_cot_va_text_gợi_ý_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(output_filename)
plt.close()

print(f"Đã lưu hình ảnh kết hợp biểu đồ cột và text tại: {output_filename}")

#nguồn csv, file heatmap trên mạng thêm vô để train thử