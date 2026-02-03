# Pandas Basic Lab - Báo Cáo Bài Tập

## Tổng Quan
Repository này chứa các bài tập cơ bản về Pandas, bao gồm Series, DataFrame, indexing, slicing, groupby, merge, và xử lý missing data cho phân tích dữ liệu thực tế.

## Công Nghệ Sử Dụng

### Thư viện chính:
- **Pandas**: Thư viện xử lý và phân tích dữ liệu dạng bảng
- **NumPy**: Hỗ trợ tính toán số học và mảng

---

## Chi Tiết Các Bài Tập

### Phần 1: Pandas Series

#### 1.1. Tạo Series từ List

**Đề bài:** Tạo Pandas Series từ list các số thực.

**Cách hoạt động:**

```python
import pandas as pd

data_pd = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data_pd)
```

**Output:**
```
0    0.25
1    0.50
2    0.75
3    1.00
dtype: float64
```

**Kiến thức áp dụng:**
- Pandas Series là mảng 1 chiều có index
- Index mặc định bắt đầu từ 0
- Kiểu dữ liệu tự động xác định (float64)

---

#### 1.2. Tạo Series từ NumPy Array

**Đề bài:** Chuyển đổi NumPy array thành Pandas Series.

**Cách hoạt động:**

```python
import numpy as np

numpy_arr = np.arange(5)
data_pd = pd.Series(numpy_arr)
print(data_pd)
```

**Output:**
```
0    0
1    1
2    2
3    3
4    4
dtype: int32
```

**Kiến thức áp dụng:**
- Tích hợp giữa NumPy và Pandas
- `.values` trả về NumPy array
- `.index` trả về RangeIndex

---

#### 1.3. Custom Indexing

**Đề bài:** Sử dụng index tùy chỉnh (chữ cái, kết hợp số và chữ).

**Cách hoạt động:**

**Letter indexing:**
```python
data_pd = pd.Series([0.25, 0.5, 0.75, 1.0], 
                    index=['a', 'b', 'c', 'd'])

print(data_pd['b'])  # Output: 0.5
print(data_pd[-1])   # Output: 1.0 (position-based)
```

**Combined indexing:**
```python
index = ['a', 'b', 'c', 'd', 3]
data_pd = pd.Series([0, 1, 2, 3, 4], index=index)

print(data_pd['a'])  # Output: 0
print(data_pd[3])    # Output: 4 (index 3, không phải vị trí 3!)
```

**Lưu ý quan trọng:**
- Index tường minh (explicit) vs vị trí (implicit)
- `data_pd[3]` truy cập theo **index 3**, không phải vị trí thứ 3
- Cần cẩn thận khi kết hợp integer và string index

---

#### 1.4. Series từ Dictionary

**Đề bài:** Tạo Series từ dictionary Python.

**Cách hoạt động:**

```python
some_population_dict = {
    'Sai Gon': 11111, 
    'Vung Tau': 22222,
    'Phan Thiet': 33333,
    'Vinh Long': 44444
}

data_pd = pd.Series(some_population_dict)
print(data_pd['Vinh Long'])  # Output: 44444

# Slicing theo index
print(data_pd['Sai Gon':'Vung Tau'])
```

**Output:**
```
Sai Gon     11111
Vung Tau    22222
dtype: int64
```

**Kiến thức áp dụng:**
- Dictionary keys → Series index
- Dictionary values → Series values
- Hỗ trợ slicing theo index labels

---

#### 1.5. Series từ Scalar

**Đề bài:** Tạo Series từ giá trị đơn.

**Cách hoạt động:**

```python
data_pd = pd.Series(5, index=['a', 'b', 'c'])
print(data_pd)
```

**Output:**
```
a    5
b    5
c    5
dtype: int64
```

**Kiến thức áp dụng:**
- Scalar được broadcast cho tất cả index
- Phải chỉ định index khi tạo từ scalar

---

### Phần 2: Pandas DataFrame

#### 2.1. Tạo DataFrame từ Dictionary

**Đề bài:** Tạo DataFrame từ dictionary với nhiều cột.

**Cách hoạt động:**

```python
area_dict = {
    'California': 423967,
    'Texas': 695662,
    'New York': 141297,
    'Florida': 170312,
    'Illinois': 149995
}

pop_dict = {
    'California': 38332521,
    'Texas': 26448193,
    'New York': 19651127,
    'Florida': 19552860,
    'Illinois': 12882135
}

states = pd.DataFrame({
    'population': pop_dict,
    'area': area_dict
})

print(states)
```

**Output:**
```
            population    area
California    38332521  423967
Texas         26448193  695662
New York      19651127  141297
Florida       19552860  170312
Illinois      12882135  149995
```

**Kiến thức áp dụng:**
- DataFrame = bảng 2 chiều với nhiều cột
- Dictionary keys → Index (hàng)
- Dictionary của dictionary → DataFrame

---

#### 2.2. DataFrame Indexing và Slicing

**Đề bài:** Truy cập và cắt dữ liệu trong DataFrame.

**Cách hoạt động:**

**Truy cập cột:**
```python
# Cách 1: Dictionary-style
print(states['population'])

# Cách 2: Attribute-style
print(states.population)
```

**Truy cập hàng:**
```python
# Theo index
print(states.loc['California'])

# Theo vị trí
print(states.iloc[0])
```

**Slicing:**
```python
# Lấy 3 hàng đầu
print(states.head(3))

# Lấy nhiều cột
print(states[['population', 'area']])

# Boolean indexing
print(states[states.area > 200000])
```

**Kiến thức áp dụng:**
- `.loc[]`: indexing theo label
- `.iloc[]`: indexing theo vị trí
- `.head()`, `.tail()`: xem nhanh dữ liệu
- Boolean masking

---

#### 2.3. Thống Kê Cơ Bản

**Đề bài:** Tính toán các thống kê mô tả.

**Cách hoạt động:**

```python
# Thống kê cơ bản
print(states.describe())

# Tính toán riêng lẻ
print(states['population'].mean())
print(states['area'].max())
print(states['population'].idxmax())  # Index của giá trị lớn nhất
```

**Output describe():**
```
         population          area
count  5.000000e+00      5.000000
mean   2.297336e+07  316247.400000
std    1.042476e+07  241732.075474
min    1.288214e+07  141297.000000
25%    1.955286e+07  149995.000000
50%    1.965113e+07  170312.000000
75%    2.644819e+07  423967.000000
max    3.833252e+07  695662.000000
```

**Kiến thức áp dụng:**
- `.describe()`: thống kê tổng quan
- `.mean()`, `.median()`, `.std()`: các hàm tính toán
- `.idxmin()`, `.idxmax()`: tìm index của min/max

---

### Phần 3: Bài Tập Thực Hành - Life Expectancy Data

**Dataset:** Life Expectancy Data từ WHO (2938 hàng, 22 cột)

**Các cột chính:**
- Country, Year, Status (Developed/Developing)
- Life expectancy, Adult Mortality, infant deaths
- Alcohol, Hepatitis B, Measles, BMI, Polio, Diphtheria
- GDP, Population, Schooling, Income composition

---

#### 3.1. Kiểm Tra Missing Data

**Đề bài:** Phân tích missing data trong dataset.

**Cách hoạt động:**

```python
df = pd.read_csv('Life_Expectancy_Data.csv')

# Đếm số NaN mỗi cột
print(df.isnull().sum())

# Tính phần trăm NaN
print((df.isnull().sum() / len(df) * 100).round(2))
```

**Kết quả:**
```
Số lượng NaN:
Country                             0
Year                                0
Life expectancy                    10
Alcohol                           194
Hepatitis B                       553
GDP                               448
Population                        652
Schooling                         163
...

Phần trăm NaN:
Alcohol                             6.60%
Hepatitis B                        18.82%
GDP                                15.25%
Population                         22.19%
```

**Kiến thức áp dụng:**
- `.isnull()`: kiểm tra NaN
- `.sum()`: đếm số lượng True
- Phân tích tỷ lệ missing data

---

#### 3.2. Xử Lý Missing Data

**Đề bài:** Fill missing values bằng mean của từng cột.

**Cách hoạt động:**

```python
print(f"Trước fillna: Tổng NaN = {df.isnull().sum().sum()}")

# Chỉ fillna cho cột số
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

print(f"Sau fillna: Tổng NaN = {df.isnull().sum().sum()}")
```

**Output:**
```
Trước fillna: Tổng NaN = 2563
Sau fillna: Tổng NaN = 0
```

**Kiến thức áp dụng:**
- `.select_dtypes()`: lọc cột theo kiểu dữ liệu
- `.fillna()`: điền giá trị thay thế NaN
- `.mean()`: tính mean cho từng cột

---

#### 3.3. GroupBy theo Country

**Đề bài:** Tính tuổi thọ trung bình theo quốc gia, tìm cao/thấp nhất.

**Cách hoạt động:**

```python
# Groupby và tính mean
grouped = df.groupby('Country')['Life expectancy ']
life_exp_by_country = grouped.mean().sort_values()

# Top 10 thấp nhất
print(life_exp_by_country.head(10))

# Top 10 cao nhất
print(life_exp_by_country.tail(10))

# Sử dụng idxmin() và idxmax()
print(f"Thấp nhất: {life_exp_by_country.idxmin()}")
print(f"Cao nhất: {life_exp_by_country.idxmax()}")
```

**Kết quả:**
```
10 quốc gia tuổi thọ THẤP NHẤT:
Country
Sierra Leone                46.11
Central African Republic    48.51
Lesotho                     48.78
Angola                      49.02
Malawi                      49.89
...

10 quốc gia tuổi thọ CAO NHẤT:
Country
Canada         81.69
Norway         81.79
Australia      81.81
Spain          82.07
Italy          82.19
France         82.22
Switzerland    82.33
Iceland        82.44
Sweden         82.52
Japan          82.54
```

**Kiến thức áp dụng:**
- `.groupby()`: nhóm theo cột
- `.mean()`: tính mean cho mỗi nhóm
- `.sort_values()`: sắp xếp kết quả
- `.idxmin()`, `.idxmax()`: tìm index của giá trị min/max

---

#### 3.4. GroupBy theo Status (Developed vs Developing)

**Đề bài:** So sánh tuổi thọ giữa nước phát triển và đang phát triển.

**Cách hoạt động:**

```python
# Groupby và aggregate nhiều functions
grouped_status = df.groupby('Status')['Life expectancy ']
stats = grouped_status.aggregate(['count', 'mean', 'median', 'std', 'min', 'max'])

print(stats)

# Tính chênh lệch
developed_mean = stats.loc['Developed', 'mean']
developing_mean = stats.loc['Developing', 'mean']
diff = developed_mean - developing_mean

print(f"Developed: {developed_mean:.2f}")
print(f"Developing: {developing_mean:.2f}")
print(f"Chênh lệch: {diff:.2f} ({diff/developing_mean*100:.1f}%)")
```

**Kết quả:**
```
Thống kê:
            count   mean  median   std   min   max
Status                                            
Developed     512  79.20   79.25  3.93  69.9  89.0
Developing   2426  67.12   69.05  8.99  36.3  89.0

Developed: 79.20
Developing: 67.12
Chênh lệch: 12.08 (18.0%)
→ Có sự khác biệt rõ rệt
```

**Kiến thức áp dụng:**
- `.aggregate()`: áp dụng nhiều hàm cùng lúc
- `.loc[]`: truy cập kết quả theo index
- Phân tích so sánh giữa các nhóm

---

#### 3.5. Tạo DataFrame Mới

**Đề bài:** Tạo DataFrame mới với cột ID (country) và Noise_level (ngẫu nhiên).

**Cách hoạt động:**

```python
# Lấy unique countries
unique_countries = df['Country'].unique()

# Tạo DataFrame từ dictionary
np.random.seed(42)
df_noise = pd.DataFrame({
    'ID': unique_countries,
    'Noise_level': np.random.uniform(30, 100, size=len(unique_countries))
})

print(df_noise.head(10))
```

**Output:**
```
                    ID  Noise_level
0          Afghanistan    56.21
1              Albania    96.55
2              Algeria    81.24
3               Angola    71.91
4  Antigua and Barbuda    40.92
5            Argentina    40.92
6              Armenia    34.07
7            Australia    90.63
8              Austria    72.08
9           Azerbaijan    79.57
```

**Kiến thức áp dụng:**
- `.unique()`: lấy giá trị duy nhất
- `pd.DataFrame()`: tạo DataFrame từ dictionary
- `np.random.uniform()`: tạo số ngẫu nhiên

---

#### 3.6. Merge DataFrames

**Đề bài:** Merge df (dữ liệu gốc) với df_noise (dữ liệu mới).

**Cách hoạt động:**

```python
# Sử dụng merge với left_on, right_on
df_merged = df.merge(df_noise, left_on='Country', right_on='ID', how='left')

print(f"df: {df.shape}")           # (2938, 22)
print(f"df_noise: {df_noise.shape}") # (193, 2)
print(f"df_merged: {df_merged.shape}") # (2938, 24)

# Xem kết quả
cols = ['Country', 'Year', 'Status', 'Life expectancy ', 'ID', 'Noise_level']
print(df_merged[cols].head(10))
```

**Output:**
```
       Country  Year      Status  Life expectancy            ID  Noise_level
0  Afghanistan  2015  Developing             65.0  Afghanistan        56.22
1  Afghanistan  2014  Developing             59.9  Afghanistan        56.22
2  Afghanistan  2013  Developing             59.9  Afghanistan        56.22
...
```

**Kiến thức áp dụng:**
- `.merge()`: nối 2 DataFrame
- `left_on`, `right_on`: chỉ định cột join
- `how='left'`: left join (giữ tất cả hàng của df trái)

**Lưu kết quả:**
```python
df_merged.to_csv('Life_Expectancy_Data_merged.csv', index=False)
```

---

## Tổng Kết

### Kỹ Năng Đạt Được:

1. **Pandas Series:**
   - Tạo Series từ list, NumPy array, dictionary, scalar
   - Custom indexing (số, chữ, kết hợp)
   - Truy cập dữ liệu theo index và position

2. **Pandas DataFrame:**
   - Tạo DataFrame từ dictionary, NumPy array
   - Indexing: `.loc[]`, `.iloc[]`, boolean masking
   - Slicing: `.head()`, `.tail()`, column selection

3. **Data Analysis:**
   - Thống kê mô tả: `.describe()`, `.mean()`, `.max()`, `.idxmax()`
   - Missing data: `.isnull()`, `.fillna()`
   - GroupBy: `.groupby()`, `.aggregate()`
   - Merge: `.merge()` với các tham số

4. **File I/O:**
   - Đọc CSV: `pd.read_csv()`
   - Ghi CSV: `.to_csv()`

### Ứng Dụng Thực Tế:
- Phân tích dữ liệu y tế (Life Expectancy)
- Làm sạch dữ liệu (missing data handling)
- Nhóm và tổng hợp thông tin (GroupBy)
- Kết hợp nhiều nguồn dữ liệu (Merge)
