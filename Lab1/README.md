# PyTorch Basic Lab - Báo Cáo Bài Tập

## Tổng Quan
Repository này chứa các bài tập cơ bản về PyTorch, bao gồm tính toán đạo hàm, Gradient Descent, và các thao tác tensor cơ bản.

## Công Nghệ Sử Dụng

### Thư viện chính:
- **PyTorch**: Framework deep learning chính
- **NumPy**: Xử lý mảng và dữ liệu số

---

## Chi Tiết Các Bài Tập

### BTVN 01: Tính Đạo Hàm với PyTorch

**Đề bài:** Cho hàm số y = 5x⁵ + 6x³ - 3x + 1, tính đạo hàm tại nhiều điểm.

**Cách hoạt động:**
1. Tạo tensor x với `requires_grad=True` để PyTorch theo dõi gradient
2. Định nghĩa hàm số polynomial
3. Gọi `y.backward()` để tính đạo hàm tự động
4. Truy cập `x.grad` để lấy giá trị đạo hàm

**Kết quả:**
```
Tại x = 0.0:  dy/dx = -3.0000    (cực tiểu địa phương)
Tại x = 1.0:  dy/dx = 40.0000    (tăng nhanh)
Tại x = 2.0:  dy/dx = 469.0000   (tăng rất nhanh)
```

**Ý nghĩa:** 
- Độ dốc âm → hàm giảm
- Độ dốc dương → hàm tăng
- Độ dốc = 0 → điểm cực trị

---

### BTVN 02: Gradient Descent Optimization

**Đề bài:** Tối ưu hàm y = x³ + 2x² + 5x + 1 với learning rate α = 0.1 trong 10 epochs.

**Cách hoạt động:**
1. Khởi tạo x = 2.0 với `requires_grad=True`
2. **Vòng lặp tối ưu:**
   - Tính giá trị hàm y
   - Tính gradient: `y.backward()`
   - Cập nhật tham số: `x = x - α * gradient`
   - Reset gradient: `x.grad.zero_()`

**Kết quả:**
```
Vòng 1:  x = 2.0000,    gradient = 25.0000
Vòng 2:  x = -0.5000,   gradient = 3.7500
Vòng 10: x = -4330.7476 (diverge do learning rate quá lớn)
```

**Nhận xét:**
- Learning rate 0.1 quá lớn → thuật toán không hội tụ
- Cần giảm learning rate xuống ~0.001 để hội tụ tốt hơn

---

### BTVN 03: Linear Regression từ Đầu

**Đề bài:** Xây dựng Linear Regression với dữ liệu giả lập y = 3x + 5 + noise.

**Cách hoạt động:**

**1. Chuẩn bị dữ liệu:**
```python
x = np.random.uniform(1, 10, 100)  # 100 mẫu từ 1-10
noise = np.random.normal(0, 0.5, 100)
y = 3*x + 5 + noise
```

**2. Khởi tạo tham số:**
```python
w = torch.randn(1, requires_grad=True)  # weight
b = torch.randn(1, requires_grad=True)  # bias
```

**3. Training loop (100 epochs):**
```python
for epoch in range(100):
    y_pred = w*x + b              # Forward pass
    loss = ((y_pred - y)**2).mean()  # MSE Loss
    loss.backward()                # Backward pass
    
    with torch.no_grad():
        w -= 0.01 * w.grad        # Update weights
        b -= 0.01 * b.grad        # Update bias
        w.grad.zero_()            # Reset gradients
        b.grad.zero_()
```

**Kết quả:**
```
Khởi tạo:    w = 0.5903,  b = -0.4278
Epoch 10:    w = 3.7192,  b = 0.2553,   Loss = 5.1409
Epoch 50:    w = 3.6082,  b = 0.9820,   Loss = 3.7789
Epoch 100:   w = 3.4925,  b = 1.7401,   Loss = 2.5917

So với giá trị thực: w = 3.0, b = 5.0
```

**Nhận xét:**
- Weight w hội tụ gần giá trị thực (3.49 ≈ 3.0)
- Bias b còn xa mục tiêu (1.74 vs 5.0) → cần train thêm hoặc tăng learning rate
- Loss giảm đều → mô hình đang học

---

### BTVN 04: Memory Sharing vs Memory Copy

**Đề bài:** Giải thích sự khác biệt giữa `torch.from_numpy()` và `torch.tensor()`.

**Cách hoạt động:**

**Thí nghiệm 1: `torch.from_numpy()` - Chia sẻ bộ nhớ**
```python
arr1 = np.array([1, 2, 3, 4, 5])
x1 = torch.from_numpy(arr1)
arr1[0] = 99
# Kết quả: x1 = [99, 2, 3, 4, 5]  ← ĐÃ THAY ĐỔI
```

**Thí nghiệm 2: `torch.tensor()` - Tạo bản sao**
```python
arr2 = np.arange(0, 5)
x2 = torch.tensor(arr2)
arr2[0] = 99
# Kết quả: x2 = [0, 1, 2, 3, 4]  ← KHÔNG THAY ĐỔI
```

**Kết luận:**

| Phương thức | Memory | Ưu điểm | Nhược điểm | Khi nào dùng |
|-------------|--------|---------|------------|--------------|
| `torch.from_numpy()` | Chia sẻ | Nhanh, tiết kiệm RAM | Không an toàn | Data cố định, cần tốc độ |
| `torch.tensor()` | Copy | An toàn, độc lập | Tốn RAM hơn | Data động, cần bảo toàn |

---

### BTVN 05: Tensor Operations

**Đề bài:** Thực hành tạo và reshape tensor với các phương thức khác nhau.

**Cách hoạt động:**

**1. Tạo tensor rỗng (chưa khởi tạo giá trị):**
```python
empty_tensor = torch.empty(3, 4)  # Shape: [3, 4]
# Giá trị: random garbage từ memory
```

**2. Tạo tensor zeros:**
```python
zeros_tensor = torch.zeros(2, 3, 4)  # Shape: [2, 3, 4]
# Tất cả giá trị = 0.0
```

**3. Tạo tensor ones:**
```python
ones_tensor = torch.ones(4, 3)  # Shape: [4, 3]
# Tất cả giá trị = 1.0
```

**4. Tạo tensor random:**
```python
random_tensor = torch.rand(3, 3)  # Shape: [3, 3]
# Giá trị ngẫu nhiên từ [0, 1) uniform distribution
```

**5. Reshape với `view()`:**
```python
original = torch.arange(12)  # [0, 1, 2, ..., 11]
reshaped = original.view(3, 4)
# Shape: [3, 4]
# [[0,  1,  2,  3],
#  [4,  5,  6,  7],
#  [8,  9, 10, 11]]
```

**6. Reshape với `view_as()`:**
```python
original2 = torch.arange(24)
target_shape = torch.zeros(4, 6)
reshaped_as = original2.view_as(target_shape)
# Shape tự động theo target: [4, 6]
```

**Nhận xét:**
- Tất cả các phương thức hoạt động đúng
- `view()` yêu cầu số phần tử không đổi (12 → 3×4)
- `view_as()` tiện lợi khi muốn clone shape từ tensor khác
