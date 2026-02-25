# Lab 5: ANN với PyTorch - MNIST & Cat-Dog Classification

## Tổng Quan
Repository này chứa bài thực hành về Mạng Nơ-ron Nhân tạo (Artificial Neural Network - ANN) sử dụng PyTorch. Chúng ta sẽ xây dựng mô hình ANN để giải quyết 2 bài toán phân loại:
1. **MNIST**: Phân loại chữ số viết tay (0-9) - 10 classes
2. **Cat and Dog**: Phân loại ảnh chó và mèo - 2 classes (binary classification)

---

## Công Nghệ Sử Dụng

### Thư viện chính:
- **PyTorch**: Framework deep learning để xây dựng và huấn luyện neural network
- **torchvision**: Dataset và transform cho computer vision
- **NumPy**: Hỗ trợ tính toán mảng và xử lý dữ liệu số
- **Matplotlib**: Vẽ đồ thị và trực quan hóa dữ liệu
- **PIL (Pillow)**: Xử lý ảnh


## Lý Thuyết Nền Tảng

### 1. ANN là gì?

**Định nghĩa:** 
Mạng Nơ-ron Nhân tạo (ANN) là một mô hình học máy lấy cảm hứng từ cấu trúc não bộ con người. Nó học từ dữ liệu bằng cách điều chỉnh các trọng số (weights) thông qua quá trình huấn luyện.

**Tại sao sử dụng ANN?**
- Có khả năng học các mẫu phi tuyến phức tạp
- Tự động trích xuất đặc trưng từ dữ liệu thô
- Linh hoạt - có thể áp dụng cho nhiều loại bài toán khác nhau
- Không cần định nghĩa quy tắc thủ công

**Ví dụ trong Lab:**
- **MNIST**: Nhận diện chữ số viết tay từ ảnh 28x28 pixels
- **Cat-Dog**: Phân biệt ảnh chó và mèo từ ảnh màu RGB

---

### 2. So Sánh Hai Bài Toán

| Đặc điểm | MNIST | Cat-Dog |
|----------|-------|---------|
| **Loại ảnh** | Grayscale (1 kênh) | RGB (3 kênh màu) |
| **Kích thước** | 28 × 28 = 784 pixels | 64 × 64 × 3 = 12,288 pixels |
| **Số classes** | 10 classes (chữ số 0-9) | 2 classes (cat, dog) |
| **Độ phức tạp** | Thấp (ảnh đơn giản, đen trắng) | Cao (ảnh màu, đa dạng) |
| **Loss function** | CrossEntropyLoss | CrossEntropyLoss |
| **Output activation** | Softmax (10 outputs) | Softmax (2 outputs) |
| **Accuracy kỳ vọng** | 95-98% | 65-75% (ANN) |

---

## Bài 1: MNIST Classification

### 1.1. Kiến Trúc Model

```
Input Layer (784)  →  Hidden Layer (128)  →  Output Layer (10)
     [28×28]              [ReLU]                [Softmax]
```

**Code:**
```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)  # 784 → 128
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)     # 128 → 10
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.flatten(x)    # (batch, 1, 28, 28) → (batch, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
```

**Số parameters:**
- fc1: 784 × 128 + 128 = 100,480
- fc2: 128 × 10 + 10 = 1,290
- **Total: 101,770 parameters**

### 1.2. Training

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 1.3. Kết Quả Mong Đợi

- **Training accuracy**: ~98-99%
- **Test accuracy**: ~95-98%
- **Training time**: ~2-3 phút (CPU), ~30 giây (GPU)
- Model hội tụ nhanh do dataset đơn giản

---

## Bài 2: Cat-Dog Classification


### 2.1. Kiến Trúc Model

```
Input (12,288)  →  Hidden1 (512)  →  Hidden2 (256)  →  Hidden3 (128)  →  Output (2)
  [64×64×3]          [ReLU+Dropout]    [ReLU+Dropout]      [ReLU]         [Softmax]
```

**Code:**
```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        
        # Các layers
        self.fc1 = nn.Linear(3 * 64 * 64, 512)  # 12,288 → 512
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)          # 512 → 256
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)          # 256 → 128
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(128, 2)            # 128 → 2
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x
```

**Số parameters:**
- fc1: 12,288 × 512 + 512 = 6,291,968
- fc2: 512 × 256 + 256 = 131,328
- fc3: 256 × 128 + 128 = 32,896
- fc4: 128 × 2 + 2 = 258
- **Total: 6,456,450 parameters** (lớn hơn MNIST rất nhiều!)

### 2.2. Tại Sao Thêm Dropout?

**Dropout là gì?**
- Kỹ thuật regularization để tránh overfitting
- Trong training: Ngẫu nhiên "tắt" một số neurons (30% trong code)
- Trong evaluation: Dùng tất cả neurons

**Tại sao cần Dropout cho Cat-Dog?**
1. **Dataset phức tạp hơn**: Ảnh màu, đa dạng về góc chụp, ánh sáng
2. **Nhiều parameters**: 6.4 triệu parameters dễ bị overfitting
3. **Dataset nhỏ hơn tương đối**: 8,000 ảnh train (so với 60,000 của MNIST)

### 2.4. Kết Quả Mong Đợi

- **Training accuracy**: ~75-85%
- **Test accuracy**: ~65-75%
- **Training time**: ~15-20 phút (CPU), ~2-3 phút (GPU)
- Model khó hội tụ hơn MNIST do dataset phức tạp

---

## So Sánh Kết Quả

### Bảng So Sánh

| Metric | MNIST | Cat-Dog |
|--------|-------|---------|
| Test Accuracy | 95-98% | 65-75% |
| Training Time | 2-3 phút | 15-20 phút |
| Parameters | 101,770 | 6,456,450 |
| Overfitting Risk | Thấp | Cao |
| Convergence | Nhanh, ổn định | Chậm, dao động |

### Tại Sao Cat-Dog Khó Hơn?

1. **Input phức tạp hơn:**
   - MNIST: 784 features (28×28 grayscale)
   - Cat-Dog: 12,288 features (64×64×3 RGB)

2. **Đa dạng trong dữ liệu:**
   - MNIST: Chữ số viết tay tương đối đồng nhất
   - Cat-Dog: Đa dạng về giống loài, góc chụp, ánh sáng, background

3. **ANN không phù hợp với ảnh:**
   - ANN "flatten" ảnh → mất thông tin về vị trí không gian
   - Không tận dụng được cấu trúc 2D của ảnh
   - CNN (Convolutional Neural Network) sẽ tốt hơn nhiều cho ảnh

4. **Số parameters quá lớn:**
   - 6.4 triệu parameters → dễ overfitting
   - Cần nhiều dữ liệu và regularization (Dropout)

---

## Activation Functions

### 1. ReLU (Rectified Linear Unit)

**Công thức:** `f(x) = max(0, x)`

**Đặc điểm:**
- Output: [0, ∞)
- Đơn giản, tính toán nhanh
- Tránh vanishing gradient
- **Vấn đề "dying ReLU"**: Neurons có thể "chết" nếu output luôn = 0

**Sử dụng:** Hidden layers trong cả MNIST và Cat-Dog

**Ví dụ:**
```
Input:  [-2, -1, 0, 1, 2]
Output: [ 0,  0, 0, 1, 2]
```

### 2. Softmax

**Công thức:** `f(x_i) = e^(x_i) / Σ e^(x_j)`

**Đặc điểm:**
- Chuyển logits thành phân bố xác suất
- Output sum = 1
- Mỗi giá trị trong [0, 1]

**Sử dụng:** Output layer cho multi-class classification

**Ví dụ MNIST:**
```
Logits:  [2.3, 0.5, -1.2, 0.8, ...]  (10 giá trị)
Softmax: [0.42, 0.07, 0.01, 0.09, ...] (tổng = 1)
         ↑ Class 0 có xác suất cao nhất (42%)
```

### 3. Sigmoid (Không dùng trong lab này)

**Công thức:** `f(x) = 1 / (1 + e^(-x))`

**Đặc điểm:**
- Output: [0, 1]
- Dùng cho binary classification với 1 output neuron
- Trong lab này, ta dùng Softmax với 2 outputs thay vì Sigmoid với 1 output

---

## Loss Functions

### CrossEntropyLoss

**Sử dụng:** Cả MNIST và Cat-Dog

**Công thức:** `L = -Σ y_i log(ŷ_i)`
- `y_i`: True label (one-hot encoded)
- `ŷ_i`: Predicted probability

**Đặc điểm:**
- Kết hợp Softmax + Negative Log Likelihood
- Phù hợp cho multi-class classification
- PyTorch tự động áp dụng Softmax nội bộ

**Ví dụ:**
```python
# True label: class 2
y = tensor([2])

# Predicted logits (trước Softmax)
logits = tensor([[0.1, 0.3, 2.5, 0.2, ...]])

loss = criterion(logits, y)
# Loss thấp nếu logits[2] cao (đúng class)
# Loss cao nếu logits[2] thấp (sai class)
```

**Lưu ý quan trọng:**
- **Input của CrossEntropyLoss**: Raw logits (trước Softmax)
- Vì vậy trong code thực tế, ta thường bỏ Softmax ở output layer khi dùng CrossEntropyLoss

---

## Optimizers

### Adam (Adaptive Moment Estimation)

**Sử dụng:** Cả MNIST và Cat-Dog với lr=0.001

**Đặc điểm:**
- Adaptive learning rate cho từng parameter
- Có momentum (nhớ hướng gradient trước)
- Hội tụ nhanh và ổn định
- Default choice cho hầu hết bài toán

**Hyperparameters:**
```python
optimizer = optim.Adam(model.parameters(), 
                       lr=0.001,      # learning rate
                       betas=(0.9, 0.999),  # momentum coefficients
                       eps=1e-8)      # numerical stability
```

**Công thức:**
```
m_t = β1 × m_{t-1} + (1-β1) × gradient     (momentum)
v_t = β2 × v_{t-1} + (1-β2) × gradient²    (variance)
θ = θ - lr × m_t / (√v_t + ε)
```

### Tại Sao Chọn Adam?

| So sánh | SGD | Adam |
|---------|-----|------|
| Learning rate | Cố định | Adaptive |
| Momentum | Không (mặc định) | Có |
| Hội tụ | Chậm, dao động | Nhanh, mượt |
| Tune parameters | Khó | Dễ (lr=0.001 thường work) |
| Use case | Baseline, research | Production, default |

---

## Kỹ Thuật Training

### 1. Batch Training

**Tại sao không train từng ảnh một?**
- Chậm: Phải update weights sau mỗi sample
- Gradient noise: Không ổn định
- Không tận dụng được vectorization/GPU

**Batch size trong lab:**
- MNIST: 64
- Cat-Dog: 64

**Trade-off:**
- Batch lớn: Nhanh, ổn định, nhưng cần nhiều memory
- Batch nhỏ: Chậm, gradient noise cao, nhưng có thể escape local minima

### 2. Epochs

**Epoch là gì?**
- 1 epoch = model xem qua toàn bộ training set 1 lần
- MNIST: 10 epochs
- Cat-Dog: 10 epochs

### 3. Device (CPU vs GPU)

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
images = images.to(device)
```

**Lợi ích GPU:**
- MNIST: 4-5x nhanh hơn CPU
- Cat-Dog: 10-15x nhanh hơn CPU (do model lớn)
- Quan trọng cho dataset lớn và model phức tạp

---

## Đánh Giá Model

### Metrics

**1. Accuracy:**
```python
correct = (predicted == labels).sum().item()
accuracy = 100 * correct / total
```
- MNIST: 95-98% test accuracy
- Cat-Dog: 65-75% test accuracy

**2. Loss:**
- Training loss: Phải giảm dần theo epoch
- Nếu không giảm → learning rate quá thấp hoặc model underfitting
- Nếu giảm quá nhanh rồi tăng → overfitting

### Visualization

**Đồ thị Loss:**
```python
plt.plot(train_loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**Đồ thị Accuracy:**
```python
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(test_acc_history, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
```

**Prediction Examples:**
- Hiển thị ảnh cùng với nhãn thật và nhãn dự đoán
- Màu xanh = đúng, màu đỏ = sai
- Giúp hiểu model đang nhầm lẫn ở đâu

---

## Hạn Chế của ANN cho Image Classification

### 1. Mất Thông Tin Không Gian

**Vấn đề:**
```python
x = self.flatten(x)  # (batch, 3, 64, 64) → (batch, 12288)
```
- Flatten biến ảnh 2D thành vector 1D
- Mất thông tin về vị trí và mối quan hệ giữa các pixels
- Ví dụ: Mắt mèo ở đâu, tai chó hình dạng ra sao → không còn

### 2. Quá Nhiều Parameters

**MNIST:**
- 784 inputs → 100,770 parameters → OK

**Cat-Dog:**
- 12,288 inputs → 6,456,450 parameters → Quá lớn!
- Nhiều parameters → overfitting
- Cần nhiều dữ liệu và regularization

### 3. Không Có Translation Invariance

**Vấn đề:**
- ANN học vị trí cố định của features
- Nếu mèo ở góc trái hay góc phải → ANN coi như 2 pattern khác nhau
- CNN giải quyết được vấn đề này bằng convolution

### 4. Không Tận Dụng Hierarchical Features

**CNN học từng bước:**
1. Layer đầu: Edges, corners
2. Layer giữa: Shapes, textures
3. Layer cuối: Object parts (mắt, tai, mũi)

**ANN:**
- Học tất cả cùng lúc
- Không có cấu trúc hierarchical
- Kém hiệu quả hơn

---

## Cải Tiến và Phát Triển

### 1. Sử Dụng CNN Thay Vì ANN

**CNN Architecture:**
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)
```

**Lợi ích:**
- Giữ được thông tin không gian
- Ít parameters hơn
- Accuracy cao hơn nhiều (>90% cho Cat-Dog)

### 2. Data Augmentation

```python
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),      # Lật ngang
    transforms.RandomRotation(10),          # Xoay
    transforms.ColorJitter(0.2, 0.2, 0.2), # Thay đổi màu
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

**Lợi ích:**
- Tăng đa dạng dữ liệu
- Giảm overfitting
- Model generalize tốt hơn

### 3. Transfer Learning

```python
from torchvision.models import resnet18

model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)  # Thay output layer
```

**Lợi ích:**
- Sử dụng pre-trained weights từ ImageNet
- Học nhanh hơn
- Accuracy rất cao (>95% cho Cat-Dog)

### 4. Tăng Kích Thước Ảnh

- Hiện tại: 64×64 pixels
- Nâng lên: 128×128 hoặc 224×224
- Nhiều thông tin hơn → accuracy cao hơn
- Trade-off: Cần nhiều memory và thời gian train

### 5. Ensemble Methods

- Train nhiều models với architectures khác nhau
- Kết hợp predictions (voting hoặc averaging)
- Accuracy cao hơn single model

---

## Troubleshooting

### 1. Loss Không Giảm

**Nguyên nhân:**
- Learning rate quá thấp
- Model quá đơn giản (underfitting)
- Gradient vanishing

**Giải pháp:**
- Tăng learning rate (0.001 → 0.01)
- Thêm layers/neurons
- Kiểm tra activation functions

### 2. Loss Tăng Đột Ngột

**Nguyên nhân:**
- Learning rate quá cao
- Gradient exploding

**Giải pháp:**
- Giảm learning rate (0.001 → 0.0001)
- Gradient clipping
- Batch normalization

### 3. Overfitting (Train Acc >> Test Acc)

**Nguyên nhân:**
- Model quá phức tạp
- Quá ít dữ liệu
- Train quá lâu

**Giải pháp:**
- Thêm Dropout (0.3 - 0.5)
- L1/L2 regularization
- Data augmentation
- Early stopping
- Giảm model complexity

### 4. Underfitting (Train Acc Thấp)

**Nguyên nhân:**
- Model quá đơn giản
- Learning rate quá thấp
- Quá ít epochs

**Giải pháp:**
- Thêm layers/neurons
- Tăng learning rate
- Train lâu hơn
- Remove regularization

### 5. Out of Memory (OOM)

**Nguyên nhân:**
- Batch size quá lớn
- Model quá lớn
- GPU memory không đủ

**Giải pháp:**
- Giảm batch size (64 → 32 → 16)
- Resize ảnh nhỏ hơn (224 → 128 → 64)
- Gradient accumulation
- Sử dụng mixed precision training

---

## Kết Luận

1. **Cấu trúc ANN cơ bản:**
   - Input → Hidden → Output
   - Forward pass và backward pass
   - Activation functions, loss functions, optimizers

2. **Hai bài toán phân loại:**
   - MNIST: Grayscale, 10 classes, đơn giản
   - Cat-Dog: RGB, 2 classes, phức tạp

3. **So sánh performance:**
   - MNIST: 95-98% accuracy (ANN đã đủ tốt)
   - Cat-Dog: 65-75% accuracy (ANN chưa tối ưu)

4. **Hạn chế của ANN cho ảnh:**
   - Mất thông tin không gian
   - Quá nhiều parameters
   - Không có translation invariance

### Hướng Phát Triển

1. **CNN (Convolutional Neural Network):**
   - Giữ được cấu trúc không gian của ảnh
   - Ít parameters hơn nhưng hiệu quả hơn
   - Accuracy có thể đạt >90% cho Cat-Dog

2. **Transfer Learning:**
   - Sử dụng pre-trained models (ResNet, VGG, EfficientNet)
   - Fine-tune cho bài toán cụ thể
   - Đạt accuracy >95% với ít thời gian train

3. **Advanced Techniques:**
   - Data Augmentation
   - Batch Normalization
   - Learning Rate Scheduling
   - Ensemble Methods

