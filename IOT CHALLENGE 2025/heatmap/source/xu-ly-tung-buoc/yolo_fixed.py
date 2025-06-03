import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import re

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
region_names = extract_region_names_from_py("D:\IOT CHALLENGE\smart-cart-heatmap\liveheatmap\source\contour_csv.py")

# === Bước 2: Đọc tọa độ người từ file CSV ===
df = pd.read_csv("person_coordinates_20250508_160218.csv")
coordinates = df[['X', 'Y']].values.T

# === Bước 3: Tính mật độ KDE ===
kde = gaussian_kde(coordinates)
x_grid = np.linspace(0, 640, 640)
y_grid = np.linspace(0, 480, 480)
X, Y = np.meshgrid(x_grid, y_grid)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# === Bước 4: Tính toán mật độ và lưu trữ thông tin ===
region_densities = []
for name, (x, y) in region_names.items():
    if 0 <= x < 640 and 0 <= y < 480:
        density = Z[int(y), int(x)]
        percentile_rank = (np.sum(Z <= density) / Z.size) * 100  # Tính hạng phần trăm
        region_densities.append({'name': name, 'density': density, 'percentile': percentile_rank})

# === Bước 5: Sắp xếp theo phần trăm từ cao xuống thấp ===
sorted_densities = sorted(region_densities, key=lambda item: item['percentile'], reverse=True)

# === Bước 6: Tạo và in gợi ý đã sắp xếp ===
print("=== GỢI Ý PHÂN BỐ NHÂN SỰ DỰA TRÊN MẬT ĐỘ (Sắp xếp từ cao xuống thấp) ===")
for item in sorted_densities:
    name = item['name']
    percentile = item['percentile']
    if percentile >= 80:
        print(f"- Khu vực {name} đang có mật độ rất cao (≥{percentile:.2f}%), cần tăng cường nhân viên khẩn cấp.")
    elif percentile >= 60:
        print(f"- Khu vực {name} mật độ cao (≥{percentile:.2f}%), nên bổ sung thêm nhân viên.")
    elif percentile >= 40:
        print(f"- Khu vực {name} mật độ trung bình (≥{percentile:.2f}%), cần theo dõi sát sao.")
    elif percentile >= 20:
        print(f"- Khu vực {name} mật độ thấp (≥{percentile:.2f}%), có thể điều phối nhân viên linh hoạt.")
    else:
        print(f"- Khu vực {name} mật độ rất thấp (<{percentile:.2f}%), cân nhắc điều chỉnh nhân sự hoặc trưng bày.")