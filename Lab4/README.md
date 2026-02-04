# ANN Lab với PyTorch - Báo Cáo Bài Tập

## Tổng Quan
Repository này chứa bài thực hành về Mạng Nơ-ron Nhân tạo (Artificial Neural Network - ANN) sử dụng PyTorch. Chúng ta sẽ xây dựng mô hình ANN để phân loại dữ liệu make_circles - một bài toán phân loại phi tuyến.

## Công Nghệ Sử Dụng

### Thư viện chính:
- **PyTorch**: Framework deep learning để xây dựng và huấn luyện neural network
- **NumPy**: Hỗ trợ tính toán mảng và xử lý dữ liệu số
- **Matplotlib**: Vẽ đồ thị và trực quan hóa dữ liệu
- **scikit-learn**: Tạo dữ liệu và đánh giá mô hình

---

## Lý Thuyết Nền Tảng

### 1. ANN là gì?

**Định nghĩa:** 
Mạng Nơ-ron Nhân tạo (ANN) là một mô hình học máy lấy cảm hứng từ cấu trúc não bộ con người. Nó học từ dữ liệu bằng cách điều chỉnh các trọng số (weights) thông qua quá trình huấn luyện.

**Tại sao sử dụng ANN?**
- Có khả năng học các mẫu phi tuyến phức tạp
- Tự động trích xuất đặc trưng từ dữ liệu thô
- Linh hoạt - có thể áp dụng cho nhiều loại bài toán khác nhau
- Không cần định nghĩa quy tắc thủ công

**Ví dụ thực tế:**
Trong bài lab này, chúng ta có 300 điểm dữ liệu được tạo bởi `make_circles` - các điểm tạo thành 2 vòng tròn đồng tâm (vòng trong và vòng ngoài). Một đường thẳng không thể phân tách 2 nhóm này, nhưng ANN có thể học ranh giới quyết định phi tuyến để phân loại chính xác.

---

### 2. Cấu Trúc ANN

#### 2.1. Kiến trúc cơ bản

```
Input Layer (2) → Hidden Layer (4) → Output Layer (1)
```

**Input Layer (Lớp đầu vào):**
- Nhận dữ liệu đầu vào
- Trong bài này: 2 features (tọa độ x, y của mỗi điểm)

**Hidden Layer (Lớp ẩn):**
- Xử lý và biến đổi dữ liệu
- Trong bài này: 4 nodes với activation function ReLU
- Đây là nơi mô hình "học" các đặc trưng

**Output Layer (Lớp đầu ra):**
- Cho kết quả cuối cùng
- Trong bài này: 1 node với Sigmoid → xác suất từ 0 đến 1

#### 2.2. Activation Functions (Hàm kích hoạt)

| Hàm | Công thức | Đặc điểm | Sử dụng |
|-----|-----------|----------|---------|
| **ReLU** | `f(x) = max(0, x)` | - Đơn giản, nhanh<br>- Tránh vanishing gradient<br>- Có thể bị "chết" (dying ReLU) | Hidden layers |
| **Sigmoid** | `f(x) = 1/(1+e^(-x))` | - Output [0, 1]<br>- Dễ bị vanishing gradient<br>- Phù hợp cho xác suất | Binary classification output |
| **Tanh** | `f(x) = (e^x - e^(-x))/(e^x + e^(-x))` | - Output [-1, 1]<br>- Zero-centered<br>- Cũng bị vanishing gradient | Hidden layers (ít dùng hơn ReLU) |
| **Softmax** | `f(x_i) = e^(x_i) / Σe^(x_j)` | - Output tổng = 1<br>- Cho phân bố xác suất | Multi-class classification |

**Tại sao cần activation functions?**
- Không có activation => ANN chỉ là một linear model
- Activation function thêm tính phi tuyến => giúp học các pattern phức tạp
- Ví dụ: ReLU biến đổi không gian để dễ phân tách hơn

---

## Chi Tiết Bài Thực Hành

### Phần 0: Chuẩn Bị Dữ Liệu

#### 0.1. Tạo Make Circles Dataset

**Đề bài:** Tạo tập dữ liệu hai vòng tròn đồng tâm cho bài toán phân loại phi tuyến.

**Cách hoạt động:**

```python
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# Tạo dữ liệu
X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# X: (300, 2) - tọa độ 300 điểm
# y: (300,) - nhãn 0 (vòng trong) hoặc 1 (vòng ngoài)
```

**Tham số quan trọng:**
- `n_samples=300`: Tạo 300 điểm dữ liệu
- `noise=0.05`: Thêm nhiễu nhẹ (5%) để dữ liệu không quá "sạch"
- `factor=0.5`: Tỷ lệ giữa bán kính vòng trong và vòng ngoài
- `random_state=42`: Seed để kết quả có thể tái tạo

**Trực quan hóa:**

```python
# Vẽ scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', alpha=0.6)
plt.xlabel('Feature 1 (x)')
plt.ylabel('Feature 2 (y)')
plt.legend()
plt.title('Make Circles Dataset')
plt.grid(True, alpha=0.3)
plt.show()
```

**Output:** Đồ thị hiển thị:
- Các điểm màu xanh (Class 0) tạo thành vòng tròn trong
- Các điểm màu đỏ (Class 1) tạo thành vòng tròn ngoài
- Dữ liệu không thể phân tách bằng đường thẳng

#### 0.2. Chia Train/Test Set

**Cách hoạt động:**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# X_train: (210, 2) - 70% dữ liệu để huấn luyện
# X_test: (90, 2) - 30% dữ liệu để kiểm tra
# y_train: (210,)
# y_test: (90,)
```

**Kiến thức áp dụng:**
- **Train set (70%)**: Dùng để mô hình học patterns
- **Test set (30%)**: Dùng để đánh giá mô hình trên dữ liệu chưa thấy
- Tỷ lệ 70/30 hoặc 80/20 là phổ biến
- `random_state` đảm bảo kết quả nhất quán

---

### Phần 1: Xây Dựng ANN với PyTorch

#### 1.1. Định nghĩa Model Class

**Đề bài:** Tạo class ANN kế thừa từ `nn.Module` với cấu trúc 2-4-1.

**Cách hoạt động:**

```python
import torch
import torch.nn as nn

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        
        # Định nghĩa các layers
        self.fc1 = nn.Linear(2, 4)   # Input => Hidden: 2 features => 4 nodes
        self.fc2 = nn.Linear(4, 1)   # Hidden => Output: 4 nodes => 1 output
        self.relu = nn.ReLU()        # Activation cho hidden layer
        self.sigmoid = nn.Sigmoid()  # Activation cho output layer
    
    def forward(self, x):
        """Forward pass - dữ liệu đi qua mạng như thế nào"""
        x = self.relu(self.fc1(x))   # Input => Hidden + ReLU
        x = self.sigmoid(self.fc2(x)) # Hidden => Output + Sigmoid
        return x
```

**Giải thích chi tiết:**

1. **`__init__` method:**
   - Khởi tạo các layers và activation functions
   - `nn.Linear(in_features, out_features)` tạo fully connected layer
   - `super().__init__()` gọi constructor của parent class

2. **`forward` method:**
   - Định nghĩa cách dữ liệu đi qua mạng
   - PyTorch tự động tính backward pass (gradient) từ forward pass
   - Thứ tự: fc1 => ReLU => fc2 => Sigmoid

3. **Số tham số:**
   ```
   fc1: 2 inputs × 4 outputs + 4 biases = 12 parameters
   fc2: 4 inputs × 1 output + 1 bias = 5 parameters
   Total: 17 parameters
   ```

**Kiến thức áp dụng:**
- Kế thừa `nn.Module` để tự động có backward pass
- `nn.Linear` tính `output = input @ weight^T + bias`
- ReLU giữ lại giá trị dương, set âm thành 0
- Sigmoid chuyển output thành xác suất (0-1)

#### 1.2. Chuyển đổi dữ liệu sang Tensor

**Cách hoạt động:**

```python
import torch

# Chuyển NumPy array => PyTorch Tensor
X_train = torch.FloatTensor(X_train)  # Shape: (210, 2)
y_train = torch.FloatTensor(y_train).unsqueeze(1)  # Shape: (210, 1)
X_test = torch.FloatTensor(X_test)    # Shape: (90, 2)
y_test = torch.FloatTensor(y_test).unsqueeze(1)    # Shape: (90, 1)
```

**Giải thích:**
- **`FloatTensor`**: Chuyển sang kiểu float32 (chuẩn cho deep learning)
- **`.unsqueeze(1)`**: Thêm dimension
  - Trước: `(210,)` => vector 1D
  - Sau: `(210, 1)` => matrix 2D với 1 cột
  - Lý do: Model output shape (210, 1) nên y_train cũng phải (210, 1) để tính loss

**Kiến thức áp dụng:**
- PyTorch hoạt động với Tensor, không phải NumPy array
- Shape matching rất quan trọng khi tính loss
- Tensor hỗ trợ GPU acceleration (nếu có)

---

### Phần 2: Huấn Luyện Model

#### 2.1. Định nghĩa Loss Function và Optimizer

**Cách hoạt động:**

```python
import torch.optim as optim

# Khởi tạo model
model = SimpleANN()

# Loss function: Binary Cross-Entropy
criterion = nn.BCELoss()

# Optimizer: Adam với learning rate 0.01
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

**Binary Cross-Entropy Loss (BCE):**

Công thức: `L = -[y log(ŷ) + (1-y) log(1-ŷ)]`

- `y`: Nhãn thật (0 hoặc 1)
- `ŷ`: Xác suất dự đoán (0 đến 1)
- Khi `y=1`: Loss = `-log(ŷ)` => càng gần 1 càng nhỏ
- Khi `y=0`: Loss = `-log(1-ŷ)` => càng gần 0 càng nhỏ

**Ví dụ:**
- Nếu y=1 (Class 1), ŷ=0.9 → Loss = -log(0.9) ≈ 0.105 (nhỏ - tốt)
- Nếu y=1 (Class 1), ŷ=0.1 → Loss = -log(0.1) ≈ 2.303 (lớn - tệ)

**Adam Optimizer:**
- Adaptive Moment Estimation
- Tự động điều chỉnh learning rate cho từng parameter
- Kết hợp momentum (nhớ hướng đi trước) và adaptive learning rate
- Công thức: `w_new = w_old - lr × m_t / (√v_t + ε)`
  - `m_t`: First moment (mean of gradients)
  - `v_t`: Second moment (variance of gradients)

#### 2.2. Training Loop

**Cách hoạt động:**

```python
epochs = 100

for epoch in range(epochs):
    # 1. Forward pass
    outputs = model(X_train)  # Dự đoán
    
    # 2. Tính loss
    loss = criterion(outputs, y_train)
    
    # 3. Backward pass
    optimizer.zero_grad()  # Xóa gradient cũ
    loss.backward()        # Tính gradient mới
    
    # 4. Update weights
    optimizer.step()
    
    # 5. In kết quả mỗi 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

**Output mẫu:**
```
Epoch [20/100], Loss: 0.4523
Epoch [40/100], Loss: 0.2891
Epoch [60/100], Loss: 0.1654
Epoch [80/100], Loss: 0.1023
Epoch [100/100], Loss: 0.0765
```

**Giải thích từng bước:**

1. **Forward pass**: 
   - Dữ liệu đi qua mạng => ra predictions
   - `outputs` shape: (210, 1)

2. **Tính loss**:
   - So sánh predictions vs ground truth
   - Một số scalar duy nhất đánh giá "độ sai"

3. **Backward pass**:
   - `optimizer.zero_grad()`: Xóa gradient từ iteration trước
     - Lý do: Gradient accumulate, không xóa sẽ cộng dồn
   - `loss.backward()`: Tính gradient của loss w.r.t tất cả parameters
     - PyTorch tự động dùng chain rule (backpropagation)

4. **Update weights**:
   - `optimizer.step()`: Cập nhật tất cả parameters
   - Công thức: `w = w - lr × ∂L/∂w`

**Kiến thức áp dụng:**
- **Epoch**: 1 lần đi qua toàn bộ training data
- **Batch**: Ở đây dùng full batch (toàn bộ 210 samples cùng lúc)
- Loss giảm theo epoch => mô hình đang học
- Nếu loss không giảm => có vấn đề (learning rate, architecture, data...)

---

### Phần 3: Đánh Giá Model

#### 3.1. Tính Accuracy trên Test Set

**Cách hoạt động:**

```python
# Tắt gradient computation (tiết kiệm memory)
with torch.no_grad():
    # Dự đoán trên test set
    y_pred = model(X_test)
    
    # Chuyển xác suất thành class (0 hoặc 1)
    y_pred_class = (y_pred >= 0.5).float()
    
    # Tính accuracy
    accuracy = (y_pred_class == y_test).float().mean()
    
    print(f'Test Accuracy: {accuracy.item()*100:.2f}%')
```

**Output mẫu:**
```
Test Accuracy: 94.44%
```

**Giải thích:**

1. **`torch.no_grad()`**:
   - Tắt tính toán gradient (không cần khi inference)
   - Giảm memory và tăng tốc độ

2. **Chuyển xác suất => class**:
   - Model output: xác suất từ 0 đến 1
   - Threshold 0.5: 
     - ≥ 0.5 => Class 1
     - < 0.5 => Class 0

3. **Tính accuracy**:
   - So sánh prediction vs ground truth
   - `(y_pred_class == y_test)` => tensor boolean
   - `.float()` chuyển True/False thành 1.0/0.0
   - `.mean()` tính trung bình => accuracy

**Kiến thức áp dụng:**
- Accuracy = (Số dự đoán đúng) / (Tổng số samples)
- Accuracy 94.44% = 85/90 samples dự đoán đúng
- Với dữ liệu balanced (50-50), accuracy là metric hợp lý

#### 3.2. Vẽ Decision Boundary

**Cách hoạt động:**

```python
import numpy as np

# Tạo mesh grid để vẽ decision boundary
x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Dự đoán cho mọi điểm trên grid
grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
with torch.no_grad():
    Z = model(grid_points)
    Z = (Z >= 0.5).float().reshape(xx.shape)

# Vẽ đồ thị
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
plt.scatter(X_test[y_test.squeeze()==0, 0], 
           X_test[y_test.squeeze()==0, 1], 
           c='blue', label='Class 0', edgecolors='k')
plt.scatter(X_test[y_test.squeeze()==1, 0], 
           X_test[y_test.squeeze()==1, 1], 
           c='red', label='Class 1', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title(f'Decision Boundary (Accuracy: {accuracy.item()*100:.2f}%)')
plt.show()
```

**Giải thích:**
- **Mesh grid**: Tạo lưới điểm phủ toàn bộ không gian
- **`np.c_`**: Nối 2 array thành 1 array 2D
- **`contourf`**: Vẽ vùng màu cho decision boundary
- Vùng xanh: Model dự đoán Class 0
- Vùng đỏ: Model dự đoán Class 1

**Kết quả quan sát:**
- Decision boundary có dạng vòng tròn (phi tuyến)
- Phân tách tốt giữa 2 classes
- Một số điểm bị misclassified nằm gần boundary

---

## Bài Tập Về Nhà

### Phần 1: Thay đổi cấu trúc ANN

#### Câu 1: Tăng số nodes trong Hidden Layer

**Yêu cầu:**
- Thay đổi từ 2-4-1 thành 2-8-1
- So sánh accuracy và số parameters

**Code:**

```python
class ANN_8nodes(nn.Module):
    def __init__(self):
        super(ANN_8nodes, self).__init__()
        self.fc1 = nn.Linear(2, 8)   # 8 nodes thay vì 4
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Huấn luyện và đánh giá tương tự như trên
```

**Phân tích:**
- Số parameters: `2×8+8 + 8×1+1 = 25` (nhiều hơn 17 của model gốc)
- Nhiều nodes hơn => capacity cao hơn => có thể học pattern phức tạp hơn
- Trade-off: Tốn thêm memory và computation
- Có thể overfitting nếu dữ liệu ít

#### Câu 2: Thêm Hidden Layer thứ 2

**Yêu cầu:**
- Thay đổi từ 2-4-1 thành 2-8-6-1 (2 hidden layers)

**Code:**

```python
class ANN_2hidden(nn.Module):
    def __init__(self):
        super(ANN_2hidden, self).__init__()
        self.fc1 = nn.Linear(2, 8)   # First hidden layer
        self.fc2 = nn.Linear(8, 6)   # Second hidden layer
        self.fc3 = nn.Linear(6, 1)   # Output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))   # ReLU cho hidden layer 2
        x = self.sigmoid(self.fc3(x))
        return x
```

**Phân tích:**
- Số parameters: `2×8+8 + 8×6+6 + 6×1+1 = 71`
- Mạng sâu hơn => có thể học hierarchical features
- Hidden layer 1: Học features đơn giản
- Hidden layer 2: Kết hợp features từ layer 1
- Risk: Overfitting, vanishing gradient (nếu mạng quá sâu)

---

### Phần 2: Thử nghiệm Loss Function và Optimizer

#### Câu 1: Sử dụng BCEWithLogitsLoss

**Yêu cầu:**
- Thay BCELoss bằng BCEWithLogitsLoss
- Bỏ Sigmoid ở output layer

**Code:**

```python
class ANN_NoSigmoid(nn.Module):
    def __init__(self):
        super(ANN_NoSigmoid, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        # Không có sigmoid!
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x

model = ANN_NoSigmoid()
criterion = nn.BCEWithLogitsLoss()  # Kết hợp Sigmoid + BCE
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop giống như trước
# Khi evaluation, dùng torch.sigmoid(outputs) để lấy xác suất
```

**Giải thích:**
- **BCEWithLogitsLoss** = Sigmoid + BCELoss trong 1 hàm
- Lợi ích: Numerical stability (tránh log(0) hoặc log(số rất nhỏ))
- Công thức: `loss = log(1 + exp(-y * logit))`
- Khi inference: Phải dùng `torch.sigmoid()` để chuyển logits → xác suất

#### Câu 2: Thay Adam bằng SGD

**Yêu cầu:**
- Sử dụng SGD thay vì Adam
- Giữ learning rate = 0.01

**Code:**

```python
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)

# Training loop giống như trước, chỉ thay optimizer
```

**So sánh Adam vs SGD:**

| Đặc điểm | Adam | SGD |
|----------|------|-----|
| Learning rate | Adaptive (tự động điều chỉnh) | Cố định |
| Momentum | Có (nhớ hướng đi trước) | Không (mặc định) |
| Tốc độ hội tụ | Nhanh hơn | Chậm hơn |
| Ổn định | Loss giảm mượt mà | Loss có thể dao động |
| Hyperparameters | Nhiều (β1, β2, ε) | Ít (chỉ lr) |
| Use case | Đa số các bài toán | Khi cần tính đơn giản |

---

### Phần 3: Phân tích kết quả

#### Yêu cầu:
1. Vẽ đồ thị loss theo epoch cho 3 trường hợp:
   - 2-4-1 với Adam
   - 2-8-1 với Adam
   - 2-4-1 với SGD
2. Trả lời câu hỏi:
   - Loss giảm nhanh/chậm nhất ở đâu?
   - Có trường hợp nào loss dao động? Tại sao?

**Code:**

```python
import matplotlib.pyplot as plt

def train_and_record(model, criterion, optimizer, epochs=100):
    """Huấn luyện và lưu loss history"""
    loss_history = []
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    return loss_history

# Train 3 models
model1 = SimpleANN()  # 2-4-1 Adam
loss1 = train_and_record(model1, nn.BCELoss(), optim.Adam(model1.parameters(), lr=0.01))

model2 = ANN_8nodes()  # 2-8-1 Adam
loss2 = train_and_record(model2, nn.BCELoss(), optim.Adam(model2.parameters(), lr=0.01))

model3 = SimpleANN()  # 2-4-1 SGD
loss3 = train_and_record(model3, nn.BCELoss(), optim.SGD(model3.parameters(), lr=0.01))

# Vẽ đồ thị
plt.figure(figsize=(10, 6))
plt.plot(loss1, label='2-4-1 (Adam)', linewidth=2)
plt.plot(loss2, label='2-8-1 (Adam)', linewidth=2)
plt.plot(loss3, label='2-4-1 (SGD)', linewidth=2, linestyle='--')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('So sánh Loss theo Epoch', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()
```

#### Phân tích kết quả:

**1. Mất mát giảm nhanh/chậm nhất?**

**NHANH NHẤT: 2-8-1 với Adam**
- Nhiều parameters (25 vs 17) => capacity cao hơn
- Adam tự động điều chỉnh learning rate => optimize hiệu quả
- Kết hợp: Mô hình mạnh + Optimizer tốt

**CHẬM NHẤT: 2-4-1 với SGD**
- Learning rate cố định => không thích ứng với landscape
- Không có momentum => dễ bị stuck ở plateau
- Đi "zigzag" thay vì đi thẳng về minimum

**2. Loss có dao động không? Tại sao?**

**CÓ - SGD thường dao động nhiều hơn Adam**

**Nguyên nhân:**

a) **SGD (Stochastic Gradient Descent):**
   - Update: `w = w - lr × gradient`
   - Learning rate CỐ ĐỊNH = 0.01
   - Khi gradient lớn => bước nhảy lớn => dễ vượt qua minimum
   - Không có "trí nhớ" về hướng đi trước => thay đổi đột ngột
   - Giống như đi bộ với bước nhảy cố định trên địa hình gồ ghề

b) **Adam (Adaptive Moment Estimation):**
   - Có **momentum**: Nhớ hướng đi trước => chuyển động mượt mà
   - Có **adaptive lr**: Tự động giảm tốc khi gần minimum => ổn định
   - Update phức tạp hơn:
     ```
     m_t = β1 × m_{t-1} + (1-β1) × gradient     (momentum)
     v_t = β2 × v_{t-1} + (1-β2) × gradient²    (variance)
     w = w - lr × m_t / (√v_t + ε)
     ```
   - Giống như đi xe có hệ thống giảm xóc và phanh ABS

c) **Kích thước model:**
   - Model nhỏ (2-4-1): Ít parameters => dễ optimize nhưng capacity thấp
   - Model lớn (2-8-1): Nhiều parameters => khó optimize hơn nhưng học tốt hơn nếu có optimizer tốt

**Kết luận:**
- Adam luôn ưu việt hơn SGD về độ ổn định
- Mô hình lớn + Adam = Kết quả tốt nhất
- SGD có thể cạnh tranh nếu tune learning rate và thêm momentum
- Trong thực tế: Dùng Adam cho hầu hết bài toán, SGD khi cần simplicity

---

## Phụ Lục

### A. Backpropagation - Cách ANN Học

**Backpropagation** là thuật toán tính gradient của loss w.r.t các weights, dùng chain rule.

**Ví dụ đơn giản:**

```
x → w1 → z → f(z) → ŷ → Loss
```

- Forward: `z = w1 × x`, `ŷ = sigmoid(z)`, `L = BCE(y, ŷ)`
- Backward (chain rule):
  ```
  ∂L/∂w1 = ∂L/∂ŷ × ∂ŷ/∂z × ∂z/∂w1
         = ∂L/∂ŷ × ŷ(1-ŷ) × x
  ```

**Trong PyTorch:**
- `loss.backward()` tự động tính tất cả gradients
- Lưu vào `.grad` attribute của mỗi parameter
- `optimizer.step()` dùng gradients này để update weights

### B. Chọn Learning Rate

**Learning rate quá lớn (ví dụ: 1.0):**
- Bước nhảy quá lớn => vượt qua minimum
- Loss dao động mạnh hoặc tăng
- Không bao giờ hội tụ

**Learning rate quá nhỏ (ví dụ: 0.00001):**
- Bước nhảy quá nhỏ => tiến rất chậm
- Cần rất nhiều epochs
- Dễ bị stuck ở local minimum

**Learning rate tốt (0.001 - 0.01):**
- Hội tụ nhanh và ổn định
- Loss giảm mượt mà
- Không vượt qua minimum

**Kỹ thuật nâng cao:**
- **Learning rate decay**: Giảm dần lr theo epoch
- **Learning rate scheduling**: Thay đổi lr theo pattern
- **Adaptive optimizers (Adam)**: Tự động điều chỉnh lr

### C. Overfitting vs Underfitting

**Underfitting (High bias):**
- Model quá đơn giản => không capture được pattern
- Train loss cao, test loss cao
- Giải pháp: 
  - Thêm layers/nodes
  - Train lâu hơn
  - Giảm regularization

**Overfitting (High variance):**
- Model quá phức tạp => học cả noise
- Train loss thấp, test loss cao
- Giải pháp:
  - Thêm dữ liệu
  - Dropout
  - L1/L2 regularization
  - Early stopping

**Just right:**
- Train loss thấp, test loss cũng thấp
- Generalize tốt trên dữ liệu mới

### D. So sánh Optimizers

| Optimizer | Công thức | Ưu điểm | Nhược điểm | Use case |
|-----------|-----------|---------|------------|----------|
| **SGD** | `w = w - lr×g` | Đơn giản, ít memory | Chậm, nhạy với lr | Baseline, research |
| **Momentum** | `v = γv + lr×g`<br>`w = w - v` | Vượt qua local min | Thêm 1 hyperparameter | Khi loss landscape phức tạp |
| **Adam** | `m = β₁m + (1-β₁)g`<br>`v = β₂v + (1-β₂)g²`<br>`w = w - lr×m/√v` | Adaptive lr, hội tụ nhanh | Nhiều hyperparameters | Default choice cho hầu hết bài toán |
| **RMSprop** | `v = γv + (1-γ)g²`<br>`w = w - lr×g/√v` | Adaptive lr cho từng param | Cần tune γ | RNN, khi gradient thay đổi nhanh |

