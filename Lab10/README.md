# Lab 10: LSTM với PyTorch - Dự Đoán Chuỗi Thời Gian & Dự Đoán Từ Tiếp Theo

## Tổng Quan

Repository này chứa bài thực hành về Mạng Nơ-ron Bộ Nhớ Dài-Ngắn Hạn (Long Short-Term Memory - LSTM) sử dụng PyTorch. Chúng ta sẽ xây dựng mô hình LSTM để giải quyết hai bài toán:

- **Time Series Forecasting**: Dự đoán giá trị tiếp theo của chuỗi thời gian (hàm sin có nhiễu) dựa trên cửa sổ các giá trị trước đó
- **Next Word Prediction**: Dự đoán từ tiếp theo trong câu dựa trên ngữ cảnh văn bản

---

## Công Nghệ Sử Dụng

### Thư viện chính:
- **PyTorch**: Framework deep learning để xây dựng và huấn luyện mô hình LSTM
- **NumPy**: Hỗ trợ tính toán mảng và tạo dữ liệu chuỗi thời gian
- **Matplotlib**: Vẽ đồ thị và trực quan hóa kết quả dự đoán
- **scikit-learn**: Chuẩn hóa dữ liệu với `MinMaxScaler` và tính các metric đánh giá

---

## Lý Thuyết Nền Tảng

### 1. LSTM là gì?

**Định nghĩa:**
LSTM (Long Short-Term Memory) là một biến thể đặc biệt của mạng nơ-ron hồi quy (RNN), được thiết kế để xử lý dữ liệu dạng chuỗi như văn bản, âm thanh hoặc chuỗi thời gian. So với RNN thông thường, LSTM hoạt động tốt hơn khi cần giữ thông tin qua nhiều bước thời gian, nhờ cơ chế **Cell State** và các **cổng (gates)** giúp chọn lọc thông tin nên giữ, nên cập nhật hay nên đưa ra đầu ra.

**Tại sao cần LSTM thay vì RNN thuần?**
- **Giải quyết vanishing gradient**: Cell State được cập nhật theo dạng cộng, giúp gradient truyền ổn định qua nhiều bước thời gian
- **Ghi nhớ dài hạn**: Cơ chế cổng cho phép LSTM giữ thông tin quan trọng qua chuỗi rất dài
- **Chọn lọc thông tin thông minh**: Mỗi cổng học cách quyết định thông tin nào cần giữ, bổ sung hoặc xuất ra
- **Linh hoạt hơn RNN**: Phù hợp với cả bài toán hồi quy chuỗi thời gian và xử lý ngôn ngữ tự nhiên

**Ứng dụng của LSTM:**
- Dự đoán chuỗi thời gian (giá cổ phiếu, thời tiết, tín hiệu cảm biến)
- Xử lý ngôn ngữ tự nhiên (NLP): sinh văn bản, dịch máy, phân loại cảm xúc
- Nhận diện giọng nói và âm nhạc

---

### 2. So Sánh RNN vs LSTM

| Đặc điểm | RNN (Lab 8) | LSTM (Lab 10) |
|----------|-------------|---------------|
| **Trạng thái** | Hidden State $h_t$ | Hidden State $h_t$ + Cell State $C_t$ |
| **Cơ chế cổng** | Không có | Forget / Input / Output Gate |
| **Vanishing gradient** | Dễ gặp | Được giảm thiểu nhờ Cell State |
| **Phụ thuộc dài hạn** | Kém | Tốt |
| **Số tham số** | Ít hơn | Nhiều hơn (~4× RNN cùng hidden_size) |
| **Bài toán phù hợp** | Chuỗi ngắn, đơn giản | Chuỗi dài, phức tạp, NLP |

---

## Kiến Trúc Model

### Bài 1: LSTMForecast (Dự đoán chuỗi thời gian)

Input (batch, seq_length, input_size=1)
↓ nn.LSTM(input_size=1, hidden_size, num_layers, batch_first=True)
↓ Hidden State h_t + Cell State C_t  →  (batch, seq_length, hidden_size)
↓ Lấy output bước cuối               →  (batch, hidden_size)
↓ FC (hidden_size → 1)
↓ Output (1 giá trị dự đoán)

**Code:**
```python
class LSTMForecast(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, num_layers=1):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden_dim)
        out = self.fc(out[:, -1, :])   # Lấy bước thời gian cuối
        return out
```

### Bài 2: NextWordLSTM (Dự đoán từ tiếp theo)

Input (batch, context_len)  — chỉ số nguyên
↓ nn.Embedding(vocab_size, embed_dim)
↓ (batch, context_len, embed_dim)
↓ nn.LSTM(embed_dim, hidden_dim, batch_first=True)
↓ Lấy output bước cuối  →  (batch, hidden_dim)
↓ FC (hidden_dim → vocab_size)
↓ Output (logits trên toàn bộ từ vựng)

**Code:**
```python
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(NextWordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)             # (batch, seq_len, embed_dim)
        out, _ = self.lstm(emb)             # (batch, seq_len, hidden_dim)
        logits = self.fc(out[:, -1, :])     # (batch, vocab_size)
        return logits
```

---

## Các Thành Phần Chính của LSTM

### 1. Cell State — Trạng Thái Ô Nhớ

Cell State $C_t$ là "băng chuyền" trung tâm của LSTM, mang thông tin xuyên suốt chuỗi với rất ít biến đổi. Không giống hidden state của RNN, Cell State được cập nhật theo phép **cộng**, giúp gradient truyền ổn định hơn:

$$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$$

---

### 2. Forget Gate — Cổng Quên

Quyết định mức độ giữ lại hay loại bỏ thông tin từ Cell State cũ:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- $f_t \approx 1$: giữ lại gần như toàn bộ thông tin cũ
- $f_t \approx 0$: loại bỏ phần lớn thông tin đó

---

### 3. Input Gate — Cổng Vào

Quyết định mức độ và nội dung thông tin mới được thêm vào Cell State:

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

- $i_t$: mức độ chấp nhận thông tin mới (0 → 1)
- $\tilde{C}_t$: nội dung mới đề xuất thêm vào Cell State

---

### 4. Output Gate — Cổng Ra

Quyết định phần nào của Cell State được đưa ra ngoài làm Hidden State:

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \cdot \tanh(C_t)$$

**Lưu ý:** $h_t$ đóng vai trò bộ nhớ ngắn hạn (được xuất ra và truyền sang bước tiếp theo), còn $C_t$ đóng vai trò bộ nhớ dài hạn — đây chính là ý nghĩa của tên gọi **Long Short-Term Memory**.

---

### 5. Chuẩn Bị Dữ Liệu (Sliding Window)

Dữ liệu chuỗi thời gian được tổ chức thành các cặp (X, y) theo kỹ thuật cửa sổ trượt:
```python
def create_sequences(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Ví dụ với window_size=3, data=[1, 2, 3, 4, 5]:
# X = [[1,2,3], [2,3,4]]
# y = [4, 5]
```

---

## Training

### Bài 1: Hồi quy chuỗi thời gian
```python
model     = LSTMForecast(hidden_dim=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

**Vòng lặp huấn luyện:**
```python
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X_train)          # (batch, 1)
    loss   = criterion(output, y_train)
    loss.backward()
    optimizer.step()
```

### Bài 2: Dự đoán từ tiếp theo
```python
model     = NextWordLSTM(vocab_size=vocab_size, embed_dim=16, hidden_dim=32)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

**Vòng lặp huấn luyện:**
```python
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X_data)           # (batch, vocab_size)
    loss   = criterion(output, y_data)
    loss.backward()
    optimizer.step()
```

### Kết Quả Mong Đợi

**Bài 1 — Dự đoán chuỗi thời gian:**

| Cấu hình | MSE | MAE |
|----------|-----|-----|
| window=5, hidden=32, lr=0.01, epochs=100 | ~0.001–0.003 | ~0.025–0.045 |
| window=3 (ngắn) | Cao hơn | Cao hơn |
| window=10 (dài) | Tương đương hoặc thấp hơn | — |
| hidden=64 | Thấp hơn nhưng chậm hơn | — |

- **Training time**: ~30 giây – 2 phút (CPU, 100 epochs)

**Bài 2 — Dự đoán từ tiếp theo:**
- Loss CrossEntropy giảm dưới 0.5 sau ~150 epochs với tập dữ liệu nhỏ
- Mô hình dự đoán đúng từ phổ biến nhất theo ngữ cảnh huấn luyện

---

## Trực Quan Hóa

### 1. Biểu Đồ Loss
```python
plt.plot(loss_history)
plt.title("Loss qua các epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss / CrossEntropy Loss")
plt.grid(True)
plt.show()
```

### 2. Dự Đoán vs Thực Tế (Bài 1)
```python
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test).numpy()

y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test.numpy())

plt.plot(y_true, label="Giá trị thực tế", linewidth=2)
plt.plot(y_pred, label="Giá trị dự đoán", linestyle="--", linewidth=2)
plt.legend()
plt.grid(True)
plt.show()
```

### 3. Dự Đoán Từ Tiếp Theo (Bài 2)
```python
def predict_next_word(model, context, word2idx, idx2word, context_len=2, top_k=3):
    tokens    = context.strip().split()
    input_ids = [word2idx.get(w, 0) for w in tokens[-context_len:]]
    x         = torch.tensor([input_ids], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1).squeeze()

    top_probs, top_indices = torch.topk(probs, k=top_k)
    for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), 1):
        print(f"  {rank}. '{idx2word[idx]}'  (xác suất: {prob:.4f})")
```

---

## Troubleshooting

### 1. Lỗi kích thước tensor (size mismatch)

**Nguyên nhân:** Quên thêm chiều `input_size` khi đưa dữ liệu vào LSTM.

**Giải pháp:**
```python
# Sai: X shape là (batch, seq)
X_batch = X_batch.to(device)

# Đúng: X shape phải là (batch, seq, input_size)
X_batch = X_batch.unsqueeze(-1).to(device)
```

### 2. Loss Không Giảm

**Nguyên nhân:** Learning rate quá nhỏ, dữ liệu chưa được chuẩn hóa hoặc số epoch chưa đủ.

**Giải pháp:**
- Tăng learning rate (0.001 → 0.01)
- Kiểm tra dữ liệu có được chuẩn hóa với MinMaxScaler chưa
- Tăng số epoch

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
- Thêm Dropout trong tầng LSTM (`dropout=0.3`)
- Giảm `hidden_dim` hoặc `num_layers`
- Giảm số epochs, dùng Early Stopping

### 5. Từ Không Có Trong Từ Điển (OOV)

**Nguyên nhân:** Từ kiểm thử chưa xuất hiện trong tập huấn luyện.

**Giải pháp:**
```python
# Dùng index mặc định (0) cho từ không biết
input_ids = [word2idx.get(w, 0) for w in tokens]
```

---

## Hướng Phát Triển

### 1. Thay LSTM bằng GRU
```python
# GRU ít tham số hơn LSTM nhưng thường đạt kết quả tương đương
self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
```

### 2. Áp dụng cho dữ liệu thực tế

- Dự đoán giá cổ phiếu (Yahoo Finance, VN-Index)
- Dự báo thời tiết (nhiệt độ, lượng mưa)
- Phân tích cảm xúc bình luận tiếng Việt

### 3. Cải thiện Training

- **Learning Rate Scheduler**: Giảm lr tự động khi val loss không cải thiện
```python
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
```
- **Early Stopping**: Dừng training khi val loss không giảm sau N epochs
- **Tăng kích thước từ điển**: Thu thập thêm văn bản để mô hình dự đoán từ chính xác hơn

### 4. Kiến Trúc Nâng Cao

- **Bidirectional LSTM**: Xử lý chuỗi theo cả hai chiều, phù hợp khi có toàn bộ chuỗi
- **Attention Mechanism**: Cho phép mô hình "tập trung" vào các bước quan trọng trong chuỗi
- **Transformer**: Kiến trúc hiện đại thay thế LSTM cho nhiều bài toán NLP và chuỗi thời gian