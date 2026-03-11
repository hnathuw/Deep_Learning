# Lab 6: CNN với PyTorch - MNIST Classification

## Tổng Quan

Repository này chứa bài thực hành về Mạng Nơ-ron Tích Chập (Convolutional Neural Network - CNN) sử dụng PyTorch. Chúng ta sẽ xây dựng mô hình CNN để giải quyết bài toán phân loại:

- **MNIST**: Phân loại chữ số viết tay (0-9) - 10 classes


---

## Công Nghệ Sử Dụng

### Thư viện chính:
- **PyTorch**: Framework deep learning để xây dựng và huấn luyện neural network
- **torchvision**: Dataset và transform cho computer vision
- **NumPy**: Hỗ trợ tính toán mảng và xử lý dữ liệu số
- **Matplotlib**: Vẽ đồ thị và trực quan hóa dữ liệu

---

## Lý Thuyết Nền Tảng

### 1. CNN là gì?

**Định nghĩa:**
CNN (Convolutional Neural Network) là một loại mạng nơ-ron nhân tạo giúp máy tính "nhìn" và hiểu ảnh, tương tự cách con người nhận diện vật thể trong đời thực. Thay vì xem toàn bộ ảnh một lúc như ANN (fully connected), CNN chia nhỏ ảnh ra, tìm các đặc trưng như đường thẳng, góc, vòng tròn, rồi ghép lại để phân loại.

**Tại sao sử dụng CNN thay vì ANN cho ảnh?**
- **Giữ được thông tin không gian**: Không flatten ảnh → không mất quan hệ vị trí giữa các pixels
- **Ít parameters hơn**: Dùng shared weights (filter) thay vì fully connected
- **Translation invariance**: Nhận diện được đặc trưng dù nằm ở bất kỳ vị trí nào trong ảnh
- **Hierarchical features**: Học từ đơn giản (cạnh, góc) đến phức tạp (hình dạng, vật thể)

---

### 2. So Sánh ANN vs CNN

| Đặc điểm | ANN (Lab 5) | CNN (Lab 6) |
|----------|-------------|-------------|
| **Xử lý ảnh** | Flatten → vector 1D | Giữ nguyên cấu trúc 2D |
| **Thông tin không gian** | Mất hoàn toàn | Được bảo toàn |
| **Số parameters (MNIST)** | 101,770 | Ít hơn nhiều |
| **Translation invariance** | Không có | Có (nhờ convolution) |
| **Hierarchical learning** | Không | Có |
| **Accuracy kỳ vọng** | 95-98% | 98-99% |

---

## Kiến Trúc Model CNN

### Cấu trúc MNIST_CNN

```
Input (1×28×28)
    ↓ Conv1 (1→16, kernel 3×3)  →  (16×26×26)
    ↓ ReLU
    ↓ MaxPool (2×2)              →  (16×13×13)
    ↓ Conv2 (16→32, kernel 3×3) →  (32×11×11)
    ↓ ReLU
    ↓ MaxPool (2×2)              →  (32×5×5)
    ↓ Flatten                    →  (800)
    ↓ FC1 (800→128)
    ↓ ReLU
    ↓ FC2 (128→10)
    ↓ Output (10 classes)
```

**Code:**
```python
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)  # 28×28 → 26×26
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # 13×13 → 11×11
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = x.view(-1, 32 * 5 * 5)               # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

## Các Thành Phần Chính của CNN

### 1. Convolutional Layer (Tầng Tích Chập)

**Cách hoạt động:**
- Một filter (kernel) nhỏ trượt qua toàn bộ ảnh
- Tại mỗi vị trí, tính tổng tích của filter với vùng ảnh tương ứng
- Tạo ra feature map thể hiện sự xuất hiện của đặc trưng đó

**Công thức kích thước output:**
```
Output size = (Input size - Kernel size + 2×Padding) / Stride + 1
```

**Ví dụ trong Lab:**
- `conv1`: Input 28×28, kernel 3×3, padding 0, stride 1 → Output 26×26
- `conv2`: Input 13×13, kernel 3×3, padding 0, stride 1 → Output 11×11

**Tham số:**
```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
# conv1: nn.Conv2d(1, 16, 3, 1, 0)   → 1 kênh vào, 16 filter ra
# conv2: nn.Conv2d(16, 32, 3, 1, 0)  → 16 kênh vào, 32 filter ra
```

---

### 2. Pooling Layer

**MaxPooling:**
- Chia feature map thành các vùng nhỏ
- Lấy giá trị lớn nhất trong mỗi vùng
- Giảm kích thước → giảm tính toán và overfitting

```python
self.pool = nn.MaxPool2d(2, 2)
# Input 26×26 → Output 13×13 (giảm 2 lần)
# Input 11×11 → Output 5×5   (giảm 2 lần, làm tròn xuống)
```

**Lợi ích:**
- Giảm kích thước không gian (downsampling)
- Tăng translation invariance
- Giảm overfitting

---

### 3. Fully Connected Layer (Sau khi Flatten)

Sau các tầng conv + pooling, ảnh đã được biến thành các đặc trưng cấp cao. Fully connected layer dùng các đặc trưng này để phân loại:

```python
x = x.view(-1, 32 * 5 * 5)   # Flatten: (batch, 32, 5, 5) → (batch, 800)
x = torch.relu(self.fc1(x))   # FC: 800 → 128
x = self.fc2(x)                # FC: 128 → 10 (logits)
```

---

## Training

### Khởi tạo

```python
model = MNIST_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Lý do chọn SGD thay vì Adam:**
- SGD với momentum hoạt động tốt với CNN cho bài toán phân loại ảnh cơ bản
- Dễ kiểm soát quá trình hội tụ bằng cách thay đổi learning rate

### Vòng lặp Huấn Luyện

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Kết Quả Mong Đợi

- **Training accuracy**: ~98-99%
- **Test accuracy**: ~98-99%
- **Training time**: ~3-5 phút (CPU, 5 epochs), ~1 phút (GPU)
- Hội tụ nhanh và ổn định hơn ANN cho cùng bài toán MNIST

---

## Trực Quan Hóa

### 1. Biểu Đồ Loss & Accuracy

```python
plt.plot(range(1, num_epochs+1), loss_values, marker='o', label='Training Loss')
plt.plot(range(1, num_epochs+1), accuracy_values, marker='o', label='Training Accuracy')
```

### 2. Kết Quả Dự Đoán

Hàm `visualize_prediction()` hiển thị 5 ảnh test kèm nhãn thật và nhãn dự đoán của model.

### 3. Feature Map Visualization

Hàm `visualize_feature_map()` trực quan hóa output của các tầng tích chập để hiểu CNN đang học gì:

```python
# Feature map từ conv1 (đặc trưng đơn giản: cạnh, nét)
conv1_output = torch.relu(model.conv1(img))

# Feature map từ conv2 (đặc trưng phức tạp hơn: hình dạng)
conv2_output = torch.relu(
    model.conv2(model.pool(torch.relu(model.conv1(img))))
)
```

**Quan sát:**
- **conv1 feature maps**: Hiển thị cạnh, đường nét, vùng sáng tối
- **conv2 feature maps**: Trừu tượng hơn, ít giống ảnh gốc, thể hiện các pattern phức tạp

---

## Bài Tập Vận Dụng

### Câu 1: Thay đổi số lượng Epoch

**Yêu cầu**: Tăng số epoch từ 5 lên 10.

**Câu hỏi:**
- Độ chính xác trên tập test có thay đổi không?
- Biểu đồ loss thay đổi ra sao?
- Có dấu hiệu overfitting không?

**Giải thích lý thuyết:**
Tăng epochs cho model thêm nhiều cơ hội học từ dữ liệu. Tuy nhiên, nếu tăng quá nhiều có thể gây overfitting (train accuracy tăng nhưng test accuracy giảm).

---

### Câu 2: Thêm một tầng tích chập

**Yêu cầu**: Thêm `conv3` vào model.

```python
self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
# Thêm vào forward: x = self.pool(torch.relu(self.conv3(x)))
```

**Lưu ý kích thước:** Sau conv3 và pooling, kích thước tensor thay đổi → cần cập nhật `fc1`:
```python
self.fc1 = nn.Linear(64 * 1 * 1, 128)  # Kiểm tra kích thước thực tế
```

**Câu hỏi:**
- Độ chính xác mới trên tập test?
- Thêm tầng conv có tác dụng gì?

---

### Câu 3: Thay đổi Learning Rate

**Yêu cầu**: Thử 2 giá trị: `lr=0.001` và `lr=0.1`.

| Learning Rate | Hành vi mong đợi |
|---------------|------------------|
| `lr=0.001` | Học chậm, hội tụ ổn định |
| `lr=0.01` (mặc định) | Cân bằng tốt |
| `lr=0.1` | Học nhanh nhưng có thể dao động mạnh, khó hội tụ |

**Câu hỏi:**
- Độ chính xác với mỗi learning rate?
- Biểu đồ loss thay đổi như thế nào?

---

### Câu 4: Vẽ Feature Map từ Conv2

**Yêu cầu**: Sửa `visualize_feature_map()` để vẽ thêm feature map từ `conv2`.

```python
def visualize_feature_map():
    model.eval()
    images, _ = next(iter(test_loader))
    img = images[0].unsqueeze(0).to(device)

    conv1_output = torch.relu(model.conv1(img))
    conv2_output = torch.relu(
        model.conv2(model.pool(torch.relu(model.conv1(img))))
    )

    plt.figure(figsize=(20, 4))
    # Subplot 1: Ảnh gốc
    # Subplot 2-3: 2 feature map từ conv1
    # Subplot 4-5: 2 feature map từ conv2
```

**Câu hỏi:**
- Feature map của conv1 và conv2 khác nhau như thế nào về mặt trực quan?
- Tại sao feature map conv2 trừu tượng hơn conv1?

---

## Optimizer

### SGD với Momentum

**Sử dụng:** `optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`

**Đặc điểm:**
- Momentum tích lũy hướng cập nhật từ các bước trước → tăng tốc hội tụ
- Ổn định hơn pure SGD, tránh dao động trong các vùng "hẹp" của loss surface

**So sánh với Adam:**

| | SGD + Momentum | Adam |
|--|--|--|
| Learning rate | Cố định | Adaptive |
| Hội tụ | Ổn định, có thể chậm | Nhanh hơn |
| Kết quả cuối | Thường tốt hơn nếu tune đúng | Ổn định với default |
| Phù hợp | CNN + classification | Nhiều loại bài toán |

---

## Đánh Giá Model

### Metrics

```python
# Test accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
```

**Kết quả kỳ vọng:**
- **Test accuracy**: ~98-99%
- **Loss cuối**: < 0.05
- **Training time**: ~3-5 phút (CPU), nhanh hơn đáng kể so với ANN cho cùng kết quả

---

## Troubleshooting

### 1. Lỗi kích thước tensor (size mismatch)

**Nguyên nhân:**
Khi thêm/bỏ tầng conv hoặc thay đổi kernel size, kích thước tensor vào fc1 thay đổi.

**Giải pháp:**
```python
# In kích thước trước fc1 để kiểm tra
x = self.pool(torch.relu(self.conv2(x)))
print(x.shape)  # Xem kích thước thực tế
x = x.view(-1, ???)  # Cập nhật số này
```

### 2. Loss Không Giảm

**Nguyên nhân:** Learning rate quá nhỏ, model chưa học được.

**Giải pháp:**
- Tăng learning rate (0.001 → 0.01)
- Kiểm tra data loading có đúng không
- Thử thêm epochs

### 3. Loss Dao Động Mạnh

**Nguyên nhân:** Learning rate quá lớn.

**Giải pháp:**
- Giảm learning rate (0.1 → 0.01)
- Thêm learning rate scheduler

### 4. Overfitting (Train Acc >> Test Acc)

**Giải pháp:**
- Thêm Dropout sau các FC layer
- Data augmentation (RandomHorizontalFlip, RandomRotation)
- Giảm số epochs

### 5. Out of Memory (OOM)

**Giải pháp:**
- Giảm batch size (64 → 32 → 16)
- Giảm số filter trong conv layers
- Dùng `torch.no_grad()` khi evaluate

---

## Hướng Phát Triển

### 1. Cải thiện kiến trúc
- Thêm Batch Normalization sau mỗi Conv layer
- Thêm Dropout trong FC layers để tránh overfitting
- Thử kiến trúc sâu hơn (3-4 conv layers)

### 2. Cải thiện training
- Learning Rate Scheduler (giảm lr theo epoch)
- Data Augmentation (tăng đa dạng dữ liệu)
- Early Stopping

### 3. Transfer Learning
```python
from torchvision.models import resnet18
model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 10)  # Fine-tune cho MNIST
```

### 4. Áp dụng cho bài toán khó hơn
- Thay MNIST bằng CIFAR-10 (ảnh màu RGB, 10 classes)
- Cat-Dog classification với CNN (accuracy >90%, so với 65-75% của ANN)
