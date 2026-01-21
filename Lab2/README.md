# NumPy Basic Lab - Báo Cáo Bài Tập

## Tổng Quan
Repository này chứa các bài tập cơ bản về NumPy, bao gồm indexing, slicing, boolean filtering, và data splitting cho Machine Learning.

## Công Nghệ Sử Dụng

### Thư viện chính:
- **NumPy**: Thư viện xử lý mảng và tính toán khoa học

---

## Chi Tiết Các Bài Tập

### BTVN 01: Trò Chơi Tic-Tac-Toe (Cờ Ca-rô 3×3)

**Đề bài:** Xây dựng trò chơi cờ ca-rô 3×3 sử dụng NumPy array với các quy tắc:
- Ma trận 3×3 ban đầu toàn số 99 (ô trống)
- X (người chơi 1) = 1, O (người chơi 2) = 0
- Luân phiên nhập tọa độ
- Kiểm tra ô đã đánh và tọa độ hợp lệ
- **Thử thách:** Tự động kiểm tra thắng khi có 3 ký tự liên tiếp

**Cách hoạt động:**

**1. Khởi tạo bàn cờ:**
```python
matrix = np.array([[99, 99, 99],
                   [99, 99, 99],
                   [99, 99, 99]])
```

**2. Hàm kiểm tra thắng:**
```python
def check_win(mat):
    # Kiểm tra hàng ngang
    for i in range(3):
        if mat[i, 0] == mat[i, 1] == mat[i, 2] != 99:
            return mat[i, 0]
    
    # Kiểm tra hàng dọc
    for j in range(3):
        if mat[0, j] == mat[1, j] == mat[2, j] != 99:
            return mat[0, j]
    
    # Kiểm tra đường chéo
    if mat[0, 0] == mat[1, 1] == mat[2, 2] != 99:
        return mat[0, 0]
    if mat[0, 2] == mat[1, 1] == mat[2, 0] != 99:
        return mat[0, 2]
    
    return None
```

**3. Game loop:**
```python
while move_count < 9:
    # Hiển thị bàn cờ
    display_board(matrix)
    
    # Nhập tọa độ
    pos = input("Nhập vị trí (hàng,cột): ")
    row, col = map(int, pos.split(','))
    
    # Kiểm tra hợp lệ
    if matrix[row, col] != 99:
        print("Ô này đã được đánh!")
        continue
    
    # Đánh dấu
    matrix[row, col] = current_player
    
    # Kiểm tra thắng
    if check_win(matrix):
        print(f"{player_name} THẮNG!")
        break
```

**Ví dụ chơi:**
```
Bàn cờ hiện tại:
- - -
- - -
- - -

Lượt của X
Nhập vị trí cho X (hàng,cột): 1,1

Bàn cờ hiện tại:
- - -
- X -
- - -

Lượt của O
Nhập vị trí cho O (hàng,cột): 0,0

...

Bàn cờ hiện tại:
O X X
X O O
- X O

 O THẮNG! 
```

**Kiến thức áp dụng:**
- NumPy array 2D indexing: `matrix[row, col]`
- Boolean comparison: `mat[i, 0] == mat[i, 1] == mat[i, 2]`
- Vòng lặp xử lý logic game
- Input validation

---

### BTVN 02: Truy Cập Phần Tử Ma Trận

**Đề bài:** Cho ma trận 3×3, thực hiện các thao tác slicing và indexing:

```python
y = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
```

**Yêu cầu:**
1. Lấy 4, 5, 6
2. Lấy 2, 5
3. Lấy 3, 4
4. Lấy 9, 6, 3

**Giải pháp:**

**1. Lấy 4, 5, 6 (toàn bộ hàng thứ 2):**
```python
# Cách 1: Slicing cả hàng
y[1, :]  # Output: [4, 5, 6]

# Cách 2: Chỉ index hàng
y[1]     # Output: [4, 5, 6]
```

**2. Lấy 2, 5 (cột giữa, 2 hàng đầu):**
```python
# Cách 1: Fancy indexing
y[[0, 1], 1]  # Output: [2, 5]

# Cách 2: Slicing
y[0:2, 1]     # Output: [2, 5]

# Cách 3: Slicing ngắn gọn
y[:2, 1]      # Output: [2, 5]
```

**3. Lấy 3, 4 (góc phải trên và góc trái giữa):**
```python
# Cách 1: Advanced indexing
y[[0, 1], [2, 0]]  # Output: [3, 4]

# Cách 2: Manual
np.array([y[0, 2], y[1, 0]])  # Output: [3, 4]
```

**4. Lấy 9, 6, 3 (cột phải, đảo ngược):**
```python
# Cách 1: Reverse slicing
y[::-1, 2]  # Output: [9, 6, 3]

# Cách 2: Fancy indexing
y[[2, 1, 0], 2]  # Output: [9, 6, 3]

# Cách 3: Index rồi reverse
y[:, 2][::-1]  # Output: [9, 6, 3]
```

---

### BTVN 03: Lọc Giá Trị Chẵn

**Đề bài:** Cho mảng x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], xuất các số chẵn bằng:
1. Vòng lặp với if
2. List comprehension

**Giải pháp:**

**Cách 1: Vòng lặp với if (Python cơ bản)**
```python
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

even_numbers_if = []
for num in x:
    if num % 2 == 0:
        even_numbers_if.append(num)

print(even_numbers_if)
# Output: [2, 4, 6, 8, 10]
```

**Cách 2: List comprehension (Python nâng cao)**
```python
even_numbers_comp = [num for num in x if num % 2 == 0]
print(even_numbers_comp)
# Output: [2, 4, 6, 8, 10]
```

---

### BTVN 04: Data Splitting cho Machine Learning

**Đề bài:** Tạo và chia dữ liệu sinh viên:
1. Tạo mảng 150×5 (chiều cao, cân nặng, tuổi, lương, điểm TB)
2. Tách X (4 cột đầu) và y (cột cuối)
3. Split 70% train / 30% test
4. Tạo 10 fold không chồng chéo (K-Fold Cross Validation)

**Cách hoạt động:**

**1. Tạo dữ liệu giả lập:**
```python
np.random.seed(42)  # Để kết quả lặp lại được

# Tạo dữ liệu ngẫu nhiên
data = np.random.rand(150, 5)

# Scale về phạm vi thực tế
data[:, 0] = data[:, 0] * 30 + 150   # Chiều cao: 150-180cm
data[:, 1] = data[:, 1] * 40 + 45    # Cân nặng: 45-85kg
data[:, 2] = data[:, 2] * 5 + 18     # Tuổi: 18-23
data[:, 3] = data[:, 3] * 15 + 5     # Lương: 5-20 triệu
data[:, 4] = data[:, 4] * 4 + 6      # Điểm: 6-10

print("Shape:", data.shape)  # (150, 5)
```

**2. Tách features và labels:**
```python
X = data[:, :-1]  # Lấy 4 cột đầu (bỏ cột cuối)
y = data[:, -1]   # Lấy cột cuối

print(f"X shape: {X.shape}")  # (150, 4)
print(f"y shape: {y.shape}")  # (150,)
```

**3. Train/Test Split (70/30):**
```python
train_size = int(0.7 * len(data))  # 105 mẫu

X_train = X[:train_size]      # 105 mẫu train
X_test = X[train_size:]       # 45 mẫu test
y_train = y[:train_size]
y_test = y[train_size:]

print(f"X_train: {X_train.shape}")  # (105, 4)
print(f"X_test: {X_test.shape}")    # (45, 4)
```

**4. K-Fold Cross Validation (10 folds):**
```python
n_folds = 10
fold_size = len(X_train) // n_folds  # 105 // 10 = 10

folds_X = []
folds_y = []

for i in range(n_folds):
    start_idx = i * fold_size
    
    # Fold cuối lấy hết phần còn lại
    if i == n_folds - 1:
        end_idx = len(X_train)
    else:
        end_idx = (i + 1) * fold_size
    
    fold_X = X_train[start_idx:end_idx]
    fold_y = y_train[start_idx:end_idx]
    
    folds_X.append(fold_X)
    folds_y.append(fold_y)
    
    print(f"Fold {i+1}: {fold_X.shape}")
```

**Kết quả:**
```
Fold 1:  (10, 4)
Fold 2:  (10, 4)
Fold 3:  (10, 4)
...
Fold 9:  (10, 4)
Fold 10: (15, 4)  ← Fold cuối chứa phần dư

Tổng: 10+10+...+10+15 = 105 
```