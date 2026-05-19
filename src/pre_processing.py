import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Processing missing data: điền bằng median
def fillna_median(df, col_list):
    df[col_list] = df[col_list].fillna(df[col_list].median())
    # Kiểm tra lại xem còn giá trị thiếu không
    print(df[col_list].isnull().sum())
    return df

# Processing missing data: điền bằng KNN
def fillna_knn(df, col_list):
    # Khởi tạo KNNImputer
    # n_neighbors=5: Tìm 5 dòng dữ liệu tương đồng nhất để tính toán giá trị điền vào
    imputer = KNNImputer(n_neighbors=5)
    # Thực hiện điền giá trị thiếu
    df[col_list] = imputer.fit_transform(df[col_list])
    # Kiểm tra lại xem còn giá trị thiếu không
    print(df[col_list].isnull().sum())
    return df

# processing missing data: specific value
def fillna_value(df, col_list, value):
    df[col_list] = df[col_list].fillna(value)
    # Kiểm tra lại xem còn giá trị thiếu không
    print(df[col_list].isnull().sum())
    return df

# processing missing data: Logistic Regression
def fillna_log(df, features, target):
    # Tách dataframe thành 2 phần: Có nhãn và Thiếu nhãn
    train_data = df[df[target].notnull()]
    predict_data = df[df[target].isnull()]

    # Khởi tạo và huấn luyện mô hình
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_data[features], train_data[target])

    # Dự đoán các giá trị bị thiếu
    predicted_values = lr.predict(predict_data[features])

    # Điền ngược lại vào DataFrame gốc
    df.loc[df[target].isnull(), target] = predicted_values

    # Kiểm tra giá trị thiếu sau khi điền
    print(df[target].isnull().sum())
    return df

# Processing outlier: sử dụng phương pháp capping
def cap_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Thay thế giá trị ngoài khoảng bằng giá trị chặn
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

# Endcoding data
def freq_encoding(df, col_list):
    for col in col_list:
        # Tính toán tần suất (tỷ lệ) xuất hiện của từng giá trị
        freq = df[col].value_counts(normalize=True)
        # Thay thế giá trị bằng tần suất tương ứng
        df[col] = df[col].map(freq)
    return df

def bi_encoding(df, col_list):
    for col in col_list:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

# Chuẩn hóa dữ liệu
#Min Max scaling
def scale_minmax(df, col_list):
    scaler = MinMaxScaler()
    df[col_list] = scaler.fit_transform(df[col_list])
    return df, scaler

#Robust scaling
def scale_robust(df, col_list):
    scaler = RobustScaler()
    df[col_list] = scaler.fit_transform(df[col_list])
    return df, scaler

#Standard scaling
def scale_standard(df, col_list):
    scaler = StandardScaler()
    df[col_list] = scaler.fit_transform(df[col_list])
    return df, scaler

def smote(X, y, k_neighbors=5):
    X_res, y_res = X.copy(), list(y)
    counts = Counter(y)
    max_count = max(counts.values())
    for label, count in counts.items():
        if count == max_count or count < 2: continue
        X_class = X[y == label]
        actual_k = min(k_neighbors, count - 1)
        knn = NearestNeighbors(n_neighbors=actual_k).fit(X_class)
        num_to_add = max_count - count
        for _ in range(num_to_add):
            idx = np.random.randint(0, count)
            neighbor_idx = np.random.choice(knn.kneighbors([X_class[idx]], return_distance=False)[0])
            diff = X_class[neighbor_idx] - X_class[idx]
            X_res = np.vstack([X_res, X_class[idx] + np.random.rand() * diff])
            y_res.append(label)
    return X_res, np.array(y_res)

def tomek_links(X, y):
    knn = NearestNeighbors(n_neighbors=2).fit(X)
    nn_idx = knn.kneighbors(X, return_distance=False)
    to_delete = []
    for i in range(len(X)):
        neighbor = nn_idx[i, 1]
        if nn_idx[neighbor, 1] == i and y[i] != y[neighbor]:
            to_delete.append(i)
    mask = np.ones(len(y), dtype=bool)
    mask[to_delete] = False
    return X[mask], y[mask]

def adasyn(X, y, k_neighbors=5):
    X_res, y_res = X.copy(), list(y)
    counts = Counter(y)
    max_count = max(counts.values())

    # Lấy nhãn lớp đa số
    majority_label = max(counts, key=counts.get)
    X_majority = X[y == majority_label]

    # Tìm KNN trên toàn bộ dữ liệu để tính độ khó
    knn_full = NearestNeighbors(n_neighbors=k_neighbors).fit(X)

    for label, count in counts.items():
        if label == majority_label or count < 2: continue

        X_class = X[y == label]
        num_to_add_total = max_count - count

        # Bước 1: Tính tỉ lệ mẫu đa số xung quanh mỗi mẫu thiểu số (độ khó)
        indices = knn_full.kneighbors(X_class, return_distance=False)
        r = []
        for idx_list in indices:
            # Đếm xem trong k hàng xóm, có bao nhiêu thằng thuộc lớp đa số
            num_majority = sum(1 for i in idx_list if y[i] == majority_label)
            r.append(num_majority / k_neighbors)

        # Chuẩn hóa r để tổng bằng 1
        r = np.array(r)
        if np.sum(r) == 0: # Nếu lớp này quá tách biệt, quay về chia đều như SMOTE
            r = np.ones(len(r)) / len(r)
        else:
            r = r / np.sum(r)

        # Bước 2: Tính số mẫu cần tạo cho từng điểm dựa trên độ khó
        g = np.round(r * num_to_add_total).astype(int)

        # Bước 3: Tạo mẫu nội suy (tương tự SMOTE)
        knn_class = NearestNeighbors(n_neighbors=min(k_neighbors, count - 1)).fit(X_class)
        for i in range(len(X_class)):
            if g[i] == 0: continue

            # Tìm hàng xóm trong cùng lớp
            neighbors = knn_class.kneighbors([X_class[i]], return_distance=False)[0]

            for _ in range(g[i]):
                neighbor_idx = np.random.choice(neighbors)
                diff = X_class[neighbor_idx] - X_class[i]
                new_sample = X_class[i] + np.random.rand() * diff
                X_res = np.vstack([X_res, new_sample])
                y_res.append(label)

    return X_res, np.array(y_res)