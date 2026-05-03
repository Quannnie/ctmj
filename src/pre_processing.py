import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

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