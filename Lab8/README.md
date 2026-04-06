# Lab 8: RNN với PyTorch - Dự Đoán Chuỗi Thời Gian

## Tổng Quan

Repository này chứa bài thực hành về Mạng Nơ-ron Hồi Quy (Recurrent Neural Network - RNN) sử dụng PyTorch. Chúng ta sẽ xây dựng mô hình RNN để giải quyết bài toán dự đoán chuỗi thời gian:

- **Sine Wave Forecasting**: Dự đoán giá trị tiếp theo của hàm sin dựa trên chuỗi các giá trị trước đó

---

## Công Nghệ Sử Dụng

### Thư viện chính:
- **PyTorch**: Framework deep learning để xây dựng và huấn luyện neural network
- **NumPy**: Hỗ trợ tính toán mảng và tạo dữ liệu chuỗi thời gian
- **Matplotlib**: Vẽ đồ thị và trực quan hóa kết quả dự đoán

---

## Lý Thuyết Nền Tảng

### 1. RNN là gì?

**Định nghĩa:**
RNN (Recurrent Neural Network) là một loại mạng nơ-ron nhân tạo được thiết kế để xử lý *dữ liệu tuần tự* (sequential data), như chuỗi thời gian, văn bản, hoặc âm thanh. Điểm đặc biệt của RNN là khả năng "ghi nhớ" thông tin từ các bước trước đó nhờ cấu trúc vòng lặp trong kiến trúc mạng.

**Tại sao cần RNN thay vì ANN/CNN cho dữ liệu tuần tự?**
- **Ghi nhớ ngữ cảnh**: Trạng thái ẩn (hidden state) lưu giữ thông tin từ các bước thời gian trước
- **Xử lý chuỗi có độ dài bất kỳ**: Không cần cố định kích thước đầu vào
- **Chia sẻ trọng số theo thời gian**: Cùng một bộ tham số được dùng lại ở mỗi bước → giảm số lượng tham số đáng kể
- **Mô hình hóa phụ thuộc dài hạn**: Có thể học các mối quan hệ giữa các phần tử ở xa nhau trong chuỗi

**Ứng dụng của RNN:**
- Dự đoán chuỗi thời gian (giá cổ phiếu, thời tiết, tín hiệu cảm biến)
- Xử lý ngôn ngữ tự nhiên (NLP): dịch máy, sinh văn bản
- Nhận diện giọng nói

---

### 2. So Sánh ANN / CNN vs RNN

| Đặc điểm | ANN (Lab 5) | CNN (Lab 6) | RNN (Lab 8) |
|----------|-------------|-------------|-------------|
| **Loại dữ liệu** | Vector cố định | Ảnh 2D | Chuỗi tuần tự |
| **Xử lý thứ tự** | Không | Không | Có |
| **Bộ nhớ ngữ cảnh** | Không | Không | Có (hidden state) |
| **Chia sẻ trọng số** | Không | Theo không gian | Theo thời gian |
| **Bài toán phù hợp** | Phân loại cơ bản | Nhận diện ảnh | Dự đoán chuỗi, NLP |

---

## Kiến Trúc Model

### Cấu trúc SimpleRNN

```
Input (batch, seq_length, input_size=1)
    ↓ nn.RNN(input_size=1, hidden_size, num_layers, batch_first=True)
    ↓ Hidden State h_t                →  (batch, seq_length, hidden_size)
    ↓ Lấy output bước cuối           →  (batch, hidden_size)
    ↓ FC (hidden_size → 1)
    ↓ Output (1 giá trị dự đoán)
```

**Code:**
```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)          # out: (batch, seq, hidden_size)
        out = self.fc(out[:, -1, :])  # Lấy output tại bước cuối
        return out
```

---

## Các Thành Phần Chính của RNN

### 1. Recurrent Layer - Tầng Hồi Quy

**Cách hoạt động:**
Tại mỗi bước thời gian $t$, RNN nhận đầu vào $x_t$ và trạng thái ẩn từ bước trước $h_{t-1}$, kết hợp để tính ra trạng thái ẩn mới $h_t$:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

Trong đó:
- $x_t$: vector đầu vào tại thời điểm $t$
- $h_{t-1}$: trạng thái ẩn tại bước trước
- $W_{xh}$: ma trận trọng số từ đầu vào đến trạng thái ẩn
- $W_{hh}$: ma trận trọng số hồi quy (trạng thái ẩn → trạng thái ẩn)
- $b_h$: bias của tầng ẩn

**Tham số:**
```python
nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
# input_size=1   : mỗi bước thời gian là 1 giá trị vô hướng
# hidden_size=32 : kích thước vector trạng thái ẩn
# num_layers=1   : số tầng RNN xếp chồng
```

---

### 2. Fully Connected Layer (Đầu ra)

Sau khi RNN xử lý toàn bộ chuỗi, chỉ lấy trạng thái ẩn tại **bước thời gian cuối cùng** để dự đoán giá trị tiếp theo:

```python
out = self.fc(out[:, -1, :])  # (batch, hidden_size) → (batch, 1)
```

**Lý do lấy bước cuối:** Trạng thái ẩn tại bước cuối đã "tích lũy" thông tin từ toàn bộ chuỗi đầu vào, phù hợp nhất để dự đoán bước tiếp theo.

---

### 3. Chuẩn Bị Dữ Liệu (Sliding Window)

Dữ liệu chuỗi thời gian được tạo thành các cặp (X, y) theo kỹ thuật cửa sổ trượt:

```python
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Ví dụ với seq_length=3, data=[1, 2, 3, 4, 5]:
# X = [[1,2,3], [2,3,4]]
# y = [4, 5]
```

---

## Training

### Khởi tạo

```python
model = SimpleRNN(input_size=1, hidden_size=32, num_layers=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

**Lý do chọn Adam:**
- Adaptive learning rate giúp hội tụ nhanh trên bài toán hồi quy chuỗi thời gian
- Phù hợp khi loss landscape không đồng đều

### Vòng lặp Huấn Luyện

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.unsqueeze(-1).to(device)  # (batch, seq, 1)
        y_batch = y_batch.unsqueeze(-1).to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Kết Quả Mong Đợi

| Cấu hình | MSE | MAE |
|----------|-----|-----|
| seq=20, hidden=32, lr=0.01, epochs=150 | ~0.0029 | ~0.044 |
| seq=10 (ngắn) | Cao hơn | Cao hơn |
| seq=30 (dài) | Tương đương hoặc cao hơn | Tương đương |
| hidden=64 | Thấp hơn nhưng chậm hơn | — |

- **Training time**: ~1-3 phút (CPU, 150 epochs)
- Hội tụ ổn định trên dữ liệu hàm sin

---

## Huấn Luyện RNN: Backpropagation Through Time (BPTT)

RNN được huấn luyện bằng thuật toán **lan truyền ngược qua thời gian** (*BPTT*). Ý tưởng là "mở" mạng RNN theo từng bước thời gian, sau đó áp dụng lan truyền ngược trên toàn bộ chuỗi:

$$L = \sum_{t=1}^{T} L_t \qquad \Rightarrow \qquad \frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}$$

### Vấn đề Gradient Biến Mất và Bùng Nổ

| Vấn đề | Biểu hiện | Giải pháp |
|--------|-----------|-----------|
| **Vanishing Gradient** | Loss không giảm, model không học được phụ thuộc dài hạn | Dùng LSTM/GRU, giảm seq_length |
| **Exploding Gradient** | Loss dao động mạnh, NaN | Gradient clipping, giảm learning rate |

Đây là lý do các kiến trúc **LSTM** và **GRU** được đề xuất để cải thiện khả năng ghi nhớ thông tin dài hạn.

---

## Trực Quan Hóa

### 1. Biểu Đồ Loss (Train vs Validation)

```python
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
```

### 2. Dự Đoán vs Thực Tế

```python
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy()

plt.plot(y_test, label='Thực tế')
plt.plot(predictions, label='Dự đoán')
plt.legend()
```

### 3. Dự Đoán Nhiều Bước (Multi-step Prediction)

```python
def multi_step_predict(model, seed_sequence, steps=3):
    preds = []
    current_seq = seed_sequence.copy()
    for _ in range(steps):
        x = torch.tensor(current_seq).float().unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            pred = model(x).item()
        preds.append(pred)
        current_seq = np.append(current_seq[1:], pred)
    return preds
```

**Lưu ý:** Sai số tích lũy theo từng bước — bước càng xa, dự đoán càng kém chính xác vì đầu vào của bước sau phụ thuộc vào dự đoán của bước trước.

---

## Bài Tập Vận Dụng

### Câu 1: Thay đổi `seq_length`

**Yêu cầu**: Thử `seq_length = 10`, `20`, `30`.

**Câu hỏi:**
- Độ chính xác (MSE, MAE) thay đổi ra sao?
- Tại sao chuỗi quá ngắn hoặc quá dài đều có thể làm giảm hiệu suất?

**Giải thích lý thuyết:**
- Chuỗi ngắn => model thiếu ngữ cảnh để học pattern
- Chuỗi dài => rủi ro vanishing gradient cao hơn, khó học phụ thuộc xa

---

### Câu 2: Thay đổi `hidden_size`

**Yêu cầu**: Thử `hidden_size = 16`, `32`, `64`.

```python
model = SimpleRNN(input_size=1, hidden_size=64, num_layers=1)
```

**Câu hỏi:**
- Hidden size lớn hơn có luôn cho kết quả tốt hơn không?
- Khi nào nên tăng hidden size?

---

### Câu 3: Thay đổi Learning Rate

**Yêu cầu**: Thử `lr = 0.001`, `0.01`, `0.05`.

| Learning Rate | Hành vi mong đợi |
|---------------|------------------|
| `lr=0.001` | Học chậm, cần nhiều epoch hơn |
| `lr=0.01` (mặc định) | Cân bằng tốt |
| `lr=0.05` | Dao động, khó hội tụ |

**Câu hỏi:**
- MSE với mỗi learning rate?
- Biểu đồ loss có dao động không?

---

### Câu 4: Thêm Tầng RNN và Dropout

**Yêu cầu**: Thêm tầng RNN thứ 2 và Dropout để chống overfitting.

```python
class DeepRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1, dropout=0.3):
        super(DeepRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout  # áp dụng giữa các tầng RNN
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
```

**Câu hỏi:**
- Mô hình sâu hơn có cải thiện MSE không?
- Dropout ảnh hưởng thế nào đến gap giữa train loss và val loss?

---

### Câu 5: Dự Đoán Nhiều Bước (Multi-step)

**Yêu cầu**: Thực hiện dự đoán 3 bước tiếp theo và so sánh với giá trị thực tế.

**Câu hỏi:**
- Sai số ở bước 1, 2, 3 thay đổi như thế nào?
- Tại sao sai số tích lũy theo số bước dự đoán?

---

## Optimizer

### Adam

**Sử dụng:** `optim.Adam(model.parameters(), lr=0.01)`

**Đặc điểm:**
- Adaptive learning rate: mỗi tham số có learning rate riêng, tự điều chỉnh theo lịch sử gradient
- Kết hợp Momentum + RMSProp → hội tụ nhanh và ổn định

**So sánh với SGD:**

| | SGD + Momentum | Adam |
|--|--|--|
| Learning rate | Cố định | Adaptive |
| Hội tụ | Ổn định, có thể chậm | Nhanh hơn |
| Phù hợp | CNN + classification | RNN + regression, NLP |
| Nhạy cảm lr | Cao | Thấp hơn |

---

## Đánh Giá Model

### Metrics

```python
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor.unsqueeze(-1).to(device)).cpu().numpy()
    actuals = y_test_tensor.numpy()

mse = mean_squared_error(actuals, preds)
mae = mean_absolute_error(actuals, preds)

print(f'MSE: {mse:.6f}')
print(f'MAE: {mae:.6f}')
```

**Kết quả kỳ vọng (cấu hình mặc định):**
- **MSE**: ~0.002–0.003
- **MAE**: ~0.042–0.050
- **Training time**: ~1–3 phút (CPU, 150 epochs)

---

## Troubleshooting

### 1. Lỗi kích thước tensor (size mismatch)

**Nguyên nhân:** Quên thêm chiều `input_size` khi đưa dữ liệu vào RNN.

**Giải pháp:**
```python
# Sai: X shape là (batch, seq)
X_batch = X_batch.to(device)

# Đúng: X shape phải là (batch, seq, input_size)
X_batch = X_batch.unsqueeze(-1).to(device)
```

### 2. Loss Không Giảm

**Nguyên nhân:** Learning rate quá nhỏ hoặc seq_length không phù hợp.

**Giải pháp:**
- Tăng learning rate (0.001 → 0.01)
- Kiểm tra dữ liệu có được chuẩn hóa (normalize) chưa
- Thêm số epoch

### 3. Loss Dao Động Mạnh

**Nguyên nhân:** Learning rate quá lớn hoặc exploding gradient.

**Giải pháp:**
- Giảm learning rate (0.05 → 0.01)
- Thêm gradient clipping:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. Overfitting (Train Loss << Val Loss)

**Giải pháp:**
- Thêm Dropout trong tầng RNN (`dropout=0.3`)
- Giảm `hidden_size` hoặc `num_layers`
- Giảm số epochs, dùng Early Stopping

### 5. Dự Đoán Multi-step Kém

**Nguyên nhân:** Sai số tích lũy do dự đoán trước dùng làm đầu vào bước sau.

**Giải pháp:**
- Tăng `seq_length` để model có ngữ cảnh tốt hơn
- Dùng kiến trúc Seq2Seq cho bài toán multi-step chuyên biệt

---

## Hướng Phát Triển

### 1. Thay RNN bằng LSTM/GRU

```python
# Thay nn.RNN bằng nn.LSTM để xử lý phụ thuộc dài hạn tốt hơn
self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

# Hoặc dùng GRU (ít tham số hơn LSTM, thường đạt kết quả tương đương)
self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
```

### 2. Áp dụng cho dữ liệu thực tế

- Dự đoán giá cổ phiếu (Yahoo Finance, VN-Index)
- Dự báo thời tiết (nhiệt độ, lượng mưa)
- Dự đoán lưu lượng truy cập website

### 3. Cải thiện Training

- **Learning Rate Scheduler**: Giảm lr tự động khi val loss không cải thiện
  ```python
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
  ```
- **Early Stopping**: Dừng training khi val loss không giảm sau N epochs
- **Batch Normalization**: Chuẩn hóa đầu vào mỗi batch

### 4. Kiến Trúc Nâng Cao

- **Bidirectional RNN**: Xử lý chuỗi theo cả hai chiều (phù hợp khi có toàn bộ chuỗi)
- **Attention Mechanism**: Cho phép model "tập trung" vào các bước quan trọng
- **Transformer**: Kiến trúc hiện đại thay thế RNN cho nhiều bài toán NLP và chuỗi thời gian