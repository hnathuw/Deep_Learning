# Deep Learning Web — ANN · CNN · RNN

Ba website Flask tích hợp mô hình Deep Learning, cho phép người dùng tải ảnh lên hoặc điều chỉnh tham số để nhận kết quả phân loại / dự đoán trực tiếp trên giao diện web.

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Chạy ứng dụng

Mở 3 terminal riêng biệt:

```bash
# Terminal 1 - ANN Web (port 5001)
cd ann-web && python app.py

# Terminal 2 - CNN Web (port 5002)
cd cnn-web && python app.py

# Terminal 3 - RNN Web (port 5003)
cd rnn-web && python app.py
```

Truy cập:
- ANN: http://localhost:5001
- CNN: http://localhost:5002
- RNN: http://localhost:5003

---

## Mô tả các Website

### 1. ANN Web — `http://localhost:5001`

**Mô tả:** Website phân loại ảnh sử dụng mạng ANN (Fully Connected Neural Network).

**Giao diện:** Dark theme, 2 tab dataset có thể chuyển đổi.

**Datasets:**
| Dataset | Classes | Input |
|---------|---------|-------|
| MNIST | Chữ số 0–9 | Ảnh xám 28×28 |
| Cats vs Dogs | Cat / Dog | Ảnh màu 64×64 |

**Luồng hoạt động:**
1. Chọn tab MNIST hoặc Cats vs Dogs
2. Kéo thả hoặc click để tải ảnh lên
3. Nhấn **PREDICT**
4. Xem kết quả + confidence score + biểu đồ xác suất từng lớp

---

### 2. CNN Web — `http://localhost:5002`

**Mô tả:** Website phân loại ảnh sử dụng mạng CNN (Convolutional Neural Network). Hỗ trợ 4 bộ dữ liệu.

**Giao diện:** Cyberpunk dark theme với grid background, accent màu cyan.

**Datasets:**
| Dataset | Classes | Input |
|---------|---------|-------|
| MNIST | Chữ số 0–9 | Ảnh xám 28×28 |
| Cats vs Dogs | Cat / Dog | Ảnh màu 160×160 |
| CIFAR-10 | 10 lớp (máy bay, ô tô...) | Ảnh 32×32 |
| PlantVillage | 38 loại bệnh lá | Ảnh 128×128 |

**Luồng hoạt động:**
1. Chọn dataset bằng card ở trên (4 nút)
2. Kéo thả hoặc click để tải ảnh lên
3. Nhấn **▶ PHÂN TÍCH**
4. Xem top-5 dự đoán với thanh xác suất

---

### 3. RNN Web — `http://localhost:5003`

**Mô tả:** Website dự đoán chuỗi thời gian (time series) sử dụng mạng RNN. Hiển thị biểu đồ tương tác.

**Giao diện:** Dark purple theme, biểu đồ Chart.js tương tác.

**Datasets:**
| Mode | Input | Output |
|------|-------|--------|
| Sin Wave (Đơn biến) | Chuỗi sin có nhiễu | Dự đoán bước tiếp theo |
| Multivariate | 3 features (sin, cos, trend) | Dự đoán target |

**Tính năng:**
- Điều chỉnh slider: số điểm, độ nhiễu, số bước tương lai
- Biểu đồ phân biệt màu: dữ liệu gốc / dự đoán / tương lai
- Thống kê tóm tắt

