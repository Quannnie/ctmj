#import package
import pandas as pd
import numpy as np
import os
import time
import warnings

#Export/ import model
import joblib

#Predifined function
from src.dataset import loader, data_info
from src.eda import data_profiling, visualizer_all_metrics, find_optimal_eps
from src.dictionaries import dictionary
from src.pre_processing import smote, tomek_links, adasyn

#Imputer
from sklearn.impute import KNNImputer

#Scaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

#Dimension reduction
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#Noise detection model
from sklearn.cluster import DBSCAN
from collections import Counter

#Clustering model
from sklearn.cluster import (KMeans, SpectralClustering, AgglomerativeClustering)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

#Prediction model
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

#Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Asking window
import tkinter as tk
from tkinter import messagebox
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', message='n_jobs value 1 overridden')

def asking_window(title, message):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    response = messagebox.askyesno(title, message)
    root.destroy()
    return response

warnings.filterwarnings('ignore', message='n_jobs value 1 overridden')

output_profiling = "../resources/profiling"

output_clustering_scaler= "../resources/model/clustering/scaler"
output_clustering_model= "../resources/model/clustering/model"
output_clustering_result = "../resources/model/clustering/result"

output_predicting_scaler= "../resources/model/predict/scaler"
output_predicting_model= "../resources/model/predict/model"
output_predicting_result= "../resources/model/predict/result"


def plot_clustering_results(viz_df, labels, title):
    """
    Vẽ 3 subplot PCA, t-SNE, UMAP tô màu theo nhãn cụm vừa tìm được.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Thiết lập màu sắc: -1 (nhiễu) sẽ có màu xám, các cụm còn lại dùng bảng màu Spectral
    unique_labels = set(labels)
    palette = sns.color_palette("Spectral", n_colors=len(unique_labels))
    if -1 in unique_labels:
        palette = {lbl: palette[i] if lbl != -1 else (0.8, 0.8, 0.8) for i, lbl in enumerate(unique_labels)}

    #PCA
    sns.scatterplot(
        ax=axes[0], x='PCA1', y='PCA2', data=viz_df,
        hue=labels, palette=palette, alpha=0.4, s=10, legend=None
    )
    axes[0].set_title('PCA - Tổng quan phương sai', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.4)

    #t-SNE
    sns.scatterplot(
        ax=axes[1], x='TSNE1', y='TSNE2', data=viz_df,
        hue=labels, palette=palette, alpha=0.4, s=10, legend=None
    )
    axes[1].set_title('t-SNE - Cụm mật độ cục bộ', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.4)

    #UMAP
    sns.scatterplot(
        ax=axes[2], x='UMAP1', y='UMAP2', data=viz_df,
        hue=labels, palette=palette, alpha=0.4, s=10, legend=None
    )
    axes[2].set_title('UMAP - Cấu trúc toàn cục & cục bộ', fontsize=14)
    axes[2].grid(True, linestyle='--', alpha=0.4)

    # Tiêu đề tổng hợp
    plt.suptitle(f'{title}\nSo sánh các phương pháp trực quan hóa cấu trúc dữ liệu', fontsize=16, y=1.05,
                 fontweight='bold')

    plt.tight_layout()
    plt.show()

def run_clustering_benchmark_with_viz(data, k_list=[2, 3, 4]):

    print("Chuẩn bị không gian trực quan (PCA, t-SNE, UMAP)...")
    pca_res = PCA(n_components=2).fit_transform(data)
    tsne_res = TSNE(n_components=2, random_state=42).fit_transform(data)
    umap_res = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(data)

    viz_df = pd.DataFrame({
        'PCA1': pca_res[:, 0], 'PCA2': pca_res[:, 1],
        'TSNE1': tsne_res[:, 0], 'TSNE2': tsne_res[:, 1],
        'UMAP1': umap_res[:, 0], 'UMAP2': umap_res[:, 1]
    })

    results = []

    def execute_and_log(name, model, params):
        start_time = time.time()
        try:
            labels = model.fit_predict(data)
            end_time = time.time()
            runtime = end_time - start_time

            counts = Counter(labels)
            n_noise = counts.pop(-1, 0)
            n_clusters = len(counts)
            cluster_sizes = {f"C{k}": v for k, v in sorted(counts.items())}

            if n_clusters > 1:
                mask = labels != -1
                s_score = silhouette_score(data[mask], labels[mask])
                dbi_score = davies_bouldin_score(data[mask], labels[mask])

                results.append({
                    "Algorithm": name, "Parameters": params, "n_clusters": n_clusters,
                    "Cluster_Sizes": cluster_sizes, "Noise_Count": n_noise,
                    "Silhouette": round(s_score, 4), "DBI": round(dbi_score, 4),
                    "Runtime (s)": round(runtime, 4)
                })

                title = f"{name} | Params: {params} | Sil: {round(s_score, 3)} | DBI: {round(dbi_score, 4)} | Noise: {n_noise}"
                plot_clustering_results(viz_df, labels, title)
                print(f"Hoàn thành {name} | Clusters: {n_clusters}")

        except Exception as e:
            print(f"Lỗi {name} ({params}): {e}")

    for k in k_list:
        models_k = {
            "K-Means": KMeans(n_clusters=k, random_state=42, n_init=10),
            "GMM": GaussianMixture(n_components=k, random_state=42),
            "Agglomerative": AgglomerativeClustering(n_clusters=k),
            "Spectral_NN_20": SpectralClustering(
            n_clusters=k, affinity='nearest_neighbors', n_neighbors=20, random_state=42),
            "Spectral_NN_50": SpectralClustering(
                n_clusters=k, affinity='nearest_neighbors', n_neighbors=50, random_state=42),
            "Spectral_RBF_high": SpectralClustering(
                n_clusters=k, affinity='rbf', gamma=1.0, random_state=42),
            "Spectral_RBF_mid": SpectralClustering(
                n_clusters=k, affinity='rbf', gamma=0.1, random_state=42),
            "Spectral_RBF_low": SpectralClustering(
                n_clusters=k, affinity='rbf', gamma=0.01, random_state=42),
            "Spectral_Laplacian": SpectralClustering(
                n_clusters=k, affinity='laplacian', random_state=42)
            }
        for name, model in models_k.items():
            execute_and_log(name, model, {"k": k})
    return pd.DataFrame(results)

def train_multi_model(data_name, X, y):
    """
    Hàm huấn luyện tất cả các mô hình trên từng biến thể dữ liệu.
    data_variants: Dictionary dạng {"Tên bộ dữ liệu": (X, y)}
    """

    models = {
        "Gradient-Boosted": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
        "Deep Learning": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=1000, random_state=42),
        "Naïve Bayes": GaussianNB(),
        "SGD Classifier": SGDClassifier(loss='hinge',class_weight='balanced', random_state=42)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }

    final_results = []

    print(f"{'='*60}")
    print(f"BẮT ĐẦU TIẾN TRÌNH HUẤN LUYỆN")
    print(f"{'='*60}")

    print(f"\n>>> Thông tin bộ dữ liệu: {data_name} | Kích thước: {X.shape}")

    for model_name, model in models.items():
        try:
            print(model_name)
            start_t = time.perf_counter()
            # Tính toán cross-validation
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring_metrics,
                error_score='raise',
                n_jobs=-1
            )
            end_t = time.perf_counter()
            total_time = end_t - start_t

            # Lưu kết quả
            r = {
                "Data Variant": data_name,
                "Model": model_name,
                "Accuracy": np.mean(cv_results['test_accuracy']),
                "Std Dev": np.std(cv_results['test_accuracy']),
                "Precision": np.mean(cv_results['test_precision']),
                "Recall": np.mean(cv_results['test_recall']),
                "F1-Score": np.mean(cv_results['test_f1']),
                "Time (s)": np.mean(cv_results['fit_time'])
            }

            final_results.append(r)
            print(f"  + {model_name.ljust(20)}: Hoàn tất (Accuracy: {r['Accuracy']:.4f} | F1: {r['F1-Score']:.4f} | Time : {total_time} )")

        except Exception as e:
            print(f"  + {model_name.ljust(20)}: LỖI - {str(e)[:50]}...")

    #XUẤT KẾT QUẢ
    df_final = pd.DataFrame(final_results)

    # Định dạng bảng
    df_display = df_final.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        df_display[col] = df_display[col].apply(lambda x: f"{x*100:.2f}%")

    # Thêm cột Std Dev vào Accuracy
    df_display['Accuracy'] = df_display['Accuracy'] + " (±" + (df_final['Std Dev']*100).map("{:.2f}%".format) + ")"
    df_display = df_display.drop(columns=['Std Dev'])

    # Sắp xếp theo F1-Score hoặc Accuracy
    df_display = df_display.sort_values(by=['Data Variant', 'F1-Score'], ascending=[True, False])

    print("\n" + "="*80)
    print("BẢNG SO SÁNH HIỆU QUẢ TỔNG THỂ")
    print("="*80)
    print(df_display.to_string(index=False))
    return df_final

def split_data_by_clusters(df, cluster_col):
    """
    Tách DataFrame gốc thành dictionary các DataFrame con theo cluster.
    """
    unique_clusters = sorted(df[cluster_col].unique())
    cluster_dict = {}
    for cid in unique_clusters:
        name = f"cluster_{cid}" if cid != -1 else "cluster_noise"
        cluster_dict[name] = df[df[cluster_col] == cid].copy()
        print(f"  + {name.ljust(15)}: {len(cluster_dict[name])} dòng")
    return cluster_dict

def preprocess_clusters(cluster_dict, target_col):
    """
    Lặp qua từng cluster để:
    1. Scale dữ liệu
    2. Lọc lớp hiếm
    3. Tạo các biến thể Resampling (SMOTE, ADASYN,...)
    """

    processed_results = {}

    print(f"\n{'='*50}")
    print(f"BƯỚC 2: SCALING & RESAMPLING CHI TIẾT")

    for cluster_name, df_raw in cluster_dict.items():
        if len(df_raw) < 10:
            print(f"  ! {cluster_name}: Bỏ qua (quá ít dữ liệu)")
            continue

        print(f"  > Đang xử lý {cluster_name}...")
        # Nhóm dùng RobustScaler
        robust_features = ['BAS_werkzaamheid_resp', 'afg_kinderen_huishouden']

        # Nhóm dùng MinMaxScaler
        minmax_features = ['GenderID', 'SPSS_Regio5', 'step2', 'step3']

        # Nhóm dùng StandardScaler
        standard_features = ['Age']

        target = 'step1'

        # XÂY DỰNG BỘ PREPROCESSOR
        preprocessor_cluster = ColumnTransformer(
            transformers=[
                ('rob', RobustScaler(), robust_features),
                ('mms', MinMaxScaler(), minmax_features),
                ('std', StandardScaler(), standard_features)
            ],
            remainder='drop'
        )

        X_scaled = preprocessor_cluster.fit_transform(df_raw)

        s2_predicting_preprocessor_file = os.path.join(output_predicting_scaler, f's2_predicting_preprocessor_{cluster_name}.pkl')
        joblib.dump(preprocessor_cluster, s2_predicting_preprocessor_file)

        y = df_raw[target_col].values

        min_samples = 6
        counts = Counter(y)
        valid_classes = [cls for cls, count in counts.items() if count >= min_samples]

        if len(valid_classes) < 2:
            print(f"    - {cluster_name}: Không đủ lớp mục tiêu để Resampling.")
            continue

        mask = np.isin(y, valid_classes)
        X_f = X_scaled[mask]
        y_f = y[mask]

        #Resampling
        try:
            X_sm, y_sm = smote(X_f, y_f)
            X_tm, y_tm = tomek_links(X_f, y_f)
            X_st, y_st = tomek_links(X_sm, y_sm)
            X_ad, y_ad = adasyn(X_f, y_f)

            # Lưu vào dictionary
            processed_results[cluster_name] = {
                "Original": (X_f, y_f),
                "SMOTE": (X_sm, y_sm),
                "Tomek": (X_tm, y_tm),
                "SMOTE-Tomek": (X_st, y_st),
                "ADASYN": (X_ad, y_ad)
            }
            print(f"{cluster_name}: Hoàn tất tạo 5 biến thể dữ liệu.")

        except Exception as e:
            print(f"{cluster_name}: Lỗi Resampling - {str(e)}")

    return processed_results

TravelDataJourneys_path = r'D:\MASTER\2.1. ADM\FINAL_PROJECT\CTMJ_ds\A_TravelDataJourneys.csv'
TravelDataUsers_path = r'D:\MASTER\2.1. ADM\FINAL_PROJECT\CTMJ_ds\B_TravelDataUsers.csv'
trigger_api = asking_window('Ask for crawling data', 'Do you want to get api data?')
if trigger_api:
    #Get data
    print("Getting data from API")
    article_id = "23690811"
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    dataset = loader.get_data(api_url)
    data_info.dataset_info(dataset)
    TravelDataUsers = dataset["B_TravelDataUsers.csv"]
    TravelDataJourneys = dataset["A_TravelDataJourneys.csv"]
else:
    try:
        TravelDataJourneys = pd.read_csv(TravelDataJourneys_path)
        TravelDataUsers = pd.read_csv(TravelDataUsers_path)
    except:
        print("Can not get data files. Back to get data from API")
        print("Getting data from API")
        article_id = "23690811"
        api_url = f"https://api.figshare.com/v2/articles/{article_id}"
        dataset = loader.get_data(api_url)
        data_info.dataset_info(dataset)
        TravelDataUsers = dataset["B_TravelDataUsers.csv"]
        TravelDataJourneys = dataset["A_TravelDataJourneys.csv"]

print('All dataset are ready for use')
#Export - có thể không run
trigger_profiling = asking_window("Ask for profiling", "Do you want to get profiling data?")

if trigger_profiling:
    # User Dashboard information
    user_output_filename = "TravelDataUsers_data_profiling.html"
    user_dashboard_title = "Travel Users Dataset Profiling"

    #Run
    data_profiling(TravelDataUsers, output_profiling ,user_output_filename, user_dashboard_title)
    print('Done profiling for TravelDataUsers')

    #Journey Dashboard information
    journey_output_filename = "TravelDataJourneys_data_profiling.html"
    journey_dashboard_title = "Travel Journeys Dataset Profiling"

    #Run
    data_profiling(TravelDataJourneys, output_profiling,journey_output_filename, journey_dashboard_title)
    print('Done profiling for TravelDataJourneys')
else:
    print('Nothing to do!')
#Preparing data for Clustering
print('Preparing data for Clustering')
dictionary = dictionary()
touchpoint_mapping = dictionary.type_touch

#Preparing data for Clustering
user_data = TravelDataUsers[~TravelDataUsers['SPSS_Regio5'].isnull()]
selected_column = [
    'UserID',
    'GenderID', #gioi tinh
    'Age', #tuoi
    'SPSS_Regio5', #vung
    'BAS_huishoudgrootte', #quy mo ho gia dinh (so nguoi trong nha)
    'BAS_werkzaamheid_resp', #nghe nghiep
    'BAS_bruto_jaarinkomen', #tong thu nhap truoc thue
    'afg_kinderen_huishouden', #so luong tre em trong gia dinh
    'AFG_sk2015', #tang lop xa hoi
    'BAS_voltooide_opleiding8_resp', #trinh do hoc van
    'SPSS_Lifestage' #giai doan cuoc doi
]

user_data = user_data[selected_column]

#Xử lý dữ liệu thiếu
imputer = KNNImputer(n_neighbors=5)
user_data_imputed_array = imputer.fit_transform(user_data)

user_data = pd.DataFrame(
    user_data_imputed_array,
    columns=user_data.columns,
    index=user_data.index
)
trigger_profiling = asking_window("Ask for profiling", "Do you want to get processed user profiling data?")
if trigger_profiling:
    processed_user_output_filename = "Processed_user_data_profiling.html"
    processed_user_dashboard_title = "Processed Travel Users Dataset Profiling"

    #Run
    data_profiling(user_data, output_profiling,processed_user_output_filename, processed_user_dashboard_title)
else:
    print('Nothing to do!')

#user_data info
print(user_data.info())
print(user_data.head())

#Preparing data for Prediction
print('Preparing data for Prediction')
#Dictionary mapping
journey_data = TravelDataJourneys[TravelDataJourneys['Duration'] > 0].copy()
#Duration > 0
journey_data['TIMESPSS'] = pd.to_datetime(journey_data['TIMESPSS'])
journey_data = journey_data.sort_values(by=['UserID', 'TIMESPSS']).reset_index(drop=True)

#Đánh dấu các click liên tiếp cùng loại của cùng 1 User
change_mask = (journey_data['type_touch'] != journey_data['type_touch'].shift()) | (journey_data['UserID'] != journey_data['UserID'].shift())
journey_data['group_id'] = change_mask.cumsum()

#Aggregation
journey_data_compressed = journey_data.groupby(['UserID', 'group_id', 'type_touch']).agg({
    'TIMESPSS': 'first',
    'Duration': 'sum',
    'PurchaseID': 'first',
    'DEVICE_TYPE': 'first',
    'purchase_own': 'max',
    'purchase_any': 'max',
    'MobilePanel': 'first',
    'FixedPanel': 'first'
}).reset_index()

#Mapping tên điểm chạm
journey_data_compressed['touchpoint_name'] = journey_data_compressed['type_touch'].map(touchpoint_mapping)

user_id_list = user_data['UserID'].unique()
journey_data_compressed = journey_data_compressed[journey_data_compressed['UserID'].isin(user_id_list)]

#Dọn dẹp và kiểm tra
journey_data_compressed = journey_data_compressed.drop(columns='group_id')

print(f"Số dòng sau khi nén: {len(journey_data_compressed)}")

#journey_data info
print(journey_data_compressed.info())
print(journey_data_compressed.head())
#Export journey data - có thể không chạy
trigger_export_data = asking_window("Ask for exporting data",
                                    "Do you want to get processed journey data?")
if trigger_export_data:
    journey_data_compressed[['UserID', 'touchpoint_name', 'TIMESPSS']].to_csv(f"{output_predicting_result}/cjm.csv", index=False)
    print(f'Journey data has been exported to {output_predicting_result}/cjm.csv')
else:
    print('Nothing to do!')

#Export journey data profiling
trigger_profiling = asking_window("Ask for profiling data",
                                 "Do you want to get processed journey profiling data?")
if trigger_profiling:
    processed_journey_output_filename = "Processed_journey_data_profiling.html"
    processed_journey_dashboard_title = "Processed Travel Journey Dataset Profiling"

    # Run
    data_profiling(user_data, output_profiling, processed_journey_output_filename, processed_journey_dashboard_title)
else:
    print('Nothing to do!')
#Cluster Modeling
#Data scaling
print('cluster Modeling')
robust_features = ['BAS_huishoudgrootte', 'BAS_werkzaamheid_resp', 'afg_kinderen_huishouden', 'BAS_voltooide_opleiding8_resp', 'SPSS_Lifestage']
minmax_features = ['GenderID', 'SPSS_Regio5', 'BAS_bruto_jaarinkomen', 'AFG_sk2015']
standard_features = ['Age']

user_data_preprocessor = ColumnTransformer(
    transformers=[
        ('rob', RobustScaler(), robust_features),
        ('mms', MinMaxScaler(), minmax_features),
        ('std', StandardScaler(), standard_features)
    ],
    remainder='drop'
)

X_scaled = user_data_preprocessor.fit_transform(user_data)

#Lưu scaler
clustering_preprocessor_file = os.path.join(output_clustering_scaler, 'user_data_preprocessor.pkl')
joblib.dump(user_data_preprocessor, clustering_preprocessor_file)
#Giảm chiều và trực quan hóa dữ liệu - có thể không chạy
print('Giảm chiều và trực quan hóa dữ liệu')
# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(X_scaled)

# UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_result = reducer.fit_transform(X_scaled)

data_viz = pd.DataFrame({
    'PCA1': pca_result[:, 0],
    'PCA2': pca_result[:, 1],
    'TSNE1': tsne_result[:, 0],
    'TSNE2': tsne_result[:, 1],
    'UMAP1': umap_result[:, 0],
    'UMAP2': umap_result[:, 1]
})

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

#PCA
sns.scatterplot(ax=axes[0], x='PCA1', y='PCA2', data=data_viz, alpha=0.4, s=10)
axes[0].set_title('PCA - Tổng quan phương sai', fontsize=14)

#t-SNE
sns.scatterplot(ax=axes[1], x='TSNE1', y='TSNE2', data=data_viz, alpha=0.4, s=10)
axes[1].set_title('t-SNE - Cụm mật độ cục bộ', fontsize=14)

#UMAP
sns.scatterplot(ax=axes[2], x='UMAP1', y='UMAP2', data=data_viz, alpha=0.4, s=10)
axes[2].set_title('UMAP - Cấu trúc toàn cục & cục bộ', fontsize=14)

plt.suptitle('So sánh các phương pháp trực quan hóa cấu trúc dữ liệu', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

#Tìm eps tối ưu cho DBSCAN
print("Tìm eps tối ưu cho DBSCAN")
k = 20
optimal_eps = find_optimal_eps(X_scaled, k)
print(f"Giá trị Eps tối ưu: {optimal_eps}")

#Model lọc nhiễu
print("Training model lọc nhiễu")
dbscan = DBSCAN(eps=optimal_eps, min_samples=k)
dbscan_labels = dbscan.fit_predict(X_scaled)
user_data['dbscan_labels'] = dbscan_labels

#Thống kê số lượng
counts = Counter(dbscan_labels)
n_noise = counts.get(-1, 0)
n_clusters = len([k for k in counts.keys() if k != -1])
total_points = len(X_scaled)
noise_percentage = (n_noise / total_points) * 100

print(f"KẾT QUẢ DBSCAN (eps={optimal_eps}) ---")
print(f"Số lượng nhiễu (Noise): {n_noise} / {total_points} điểm")
print(f"Tỷ lệ nhiễu: {noise_percentage:.2f}%")
print(f"Số lượng cụm tìm thấy: {n_clusters}")
print(f"Chi tiết các cụm: {dict(sorted(counts.items()))}")

# Tạo tập dữ liệu sạch
mask_clean = dbscan_labels != -1
X_clean = X_scaled[mask_clean]
print(f"Dữ liệu ban đầu: {len(X_scaled)}")
print(f"Dữ liệu sau khi lọc nhiễu: {len(X_clean)}")

#Lưu model
joblib.dump(dbscan, os.path.join(output_clustering_model, 'dbscan_clustering_model.pkl'))
#Có thể không chạy
#Model phân cụm
#Tìm số k
visualizer_all_metrics(X_clean)

#So sánh model

df_final = run_clustering_benchmark_with_viz(X_clean)

print("\n--- BẢNG SO SÁNH---")
print(df_final.sort_values("Silhouette", ascending=False))
df_final.to_excel(f'{output_predicting_result}/training_result.xlsx')
#Final clustering model
final_model = SpectralClustering(n_clusters=3, affinity='rbf', gamma=1.0, random_state=42)
final_labels_clean = final_model.fit_predict(X_clean)

#Mapping kết quả
s_score = silhouette_score(X_clean, final_labels_clean)
dbi_score = davies_bouldin_score(X_clean, final_labels_clean)
counts_clean = Counter(final_labels_clean)
cluster_sizes = {f"C{k}": v for k, v in sorted(counts_clean.items())}
user_data['final_label'] = -1

mask_clean = (user_data['dbscan_labels'] != -1)

user_data.loc[mask_clean, 'final_label'] = final_model.labels_

print("Thống kê nhãn")
print(user_data['final_label'].value_counts())
user_data.to_csv(os.path.join(output_clustering_result,'clustering_result.csv'))

joblib.dump(final_model, os.path.join(output_clustering_model, 'spectral_clustering_model.pkl'))
#Export labeled user_data profiling - có thể không chạy
trigger_profiling = asking_window("Asking for profiling",
                                "Do you want to get labeled user_data profiling?")
if trigger_profiling:
    labeled_user_data_output_filename = "Labeled_user_data_profiling.html"
    labeled_user_data_dashboard_title = "Labeled User Dataset Profiling"

    # Run
    data_profiling(user_data, output_profiling, labeled_user_data_output_filename, labeled_user_data_dashboard_title)
else:
    print('Nothing to do!')
#Predict modeling
#Preparing data
print('Predict modeling')
print('Preparing data')
journey_data_compressed = journey_data_compressed.sort_values(
    by=['UserID', 'TIMESPSS'],
    ascending=[True, False]
)
journey_data_compressed['step'] = journey_data_compressed.groupby('UserID').cumcount() + 1
journey_data_compressed = journey_data_compressed[journey_data_compressed['step'] <=3]
df_pivot = journey_data_compressed[['UserID', 'step', 'type_touch']]
df_step_pivot = journey_data_compressed.pivot(index='UserID', columns='step', values='type_touch')

#Đổi tên cột từ 1, 2, 3 thành step1, step2, step3
df_step_pivot.columns = [f'step{int(c)}' for c in df_step_pivot.columns]
df_final_steps = df_step_pivot.reset_index()
df_final_steps = df_final_steps[~df_final_steps['step2'].isnull() & ~df_final_steps['step3'].isnull()]

# Danh sách các cột thông tin người dùng cần lấy
selected_user_info = [
    'UserID',
    'GenderID',
    'Age',
    'SPSS_Regio5',
    'BAS_werkzaamheid_resp',
    'afg_kinderen_huishouden',
    'final_label'
]
# Mapping
final_modeling_df = df_final_steps.merge(
    user_data[selected_user_info],
    on='UserID',
    how='inner'
)
#Export data profiling - có thể không chạy
trigger_profiling = asking_window("Asking for profiling",
                                "Do you want to get processed journey data for modeling profiling?")
if trigger_profiling:
    processed_journey_data_for_modeling_output_filename = "processed_journey_data_for_modeling_profiling.html"
    processed_journey_data_for_modeling_dashboard_title = "Processed journey data for modeling Profiling"

    # Run
    data_profiling(user_data, output_profiling, processed_journey_data_for_modeling_output_filename,
                   processed_journey_data_for_modeling_dashboard_title)
else:
    print('Nothing to do!')
#Senario 1: Using all data, include cluster information to train predict model
#Data scaling
# Nhóm dùng RobustScaler
robust_features = ['BAS_werkzaamheid_resp', 'afg_kinderen_huishouden']

# Nhóm dùng MinMaxScaler
minmax_features = ['GenderID', 'SPSS_Regio5', 'final_label', 'step2', 'step3'] #Include clustering information

# Nhóm dùng StandardScaler
standard_features = ['Age']

target = 'step1'

preprocessor_s1 = ColumnTransformer(
    transformers=[
        ('rob', RobustScaler(), robust_features),
        ('mms', MinMaxScaler(), minmax_features),
        ('std', StandardScaler(), standard_features)
    ],
    remainder='drop'
)

X_scaled_raw = preprocessor_s1.fit_transform(final_modeling_df)
s1_predicting_preprocessor_file = os.path.join(output_predicting_scaler, 's1_predicting_preprocessor.pkl')
joblib.dump(preprocessor_s1, s1_predicting_preprocessor_file)

# Lấy nhãn mục tiêu (Target)
y_raw = final_modeling_df[target].values

#LỌC LỚP HIẾM (Để SMOTE và CV không bị lỗi)
#Lọc dựa trên y_raw để đảm bảo mọi lớp đều có ít nhất 6 mẫu
min_samples = 6
counts = Counter(y_raw)
valid_classes = [cls for cls, count in counts.items() if count >= min_samples]

# Tạo mask để lọc đồng thời X, y
mask = np.isin(y_raw, valid_classes)

X_filtered = X_scaled_raw[mask]
y_filtered = y_raw[mask]

#thông tin
print(f"Dữ liệu sẵn sàng")
print(f"--- Tổng số mẫu sau lọc: {len(y_filtered)}")
print(f"--- Số lượng features: {X_filtered.shape[1]}")
print(f"--- Danh sách lớp mục tiêu hợp lệ: {valid_classes}")

#Resampling
print("Đang tiền xử lý Resampling...")
X_sm, y_sm = smote(X_filtered, y_filtered)
X_tm, y_tm = tomek_links(X_filtered, y_filtered)
X_st, y_st = tomek_links(X_sm, y_sm)
X_ad, y_ad = adasyn(X_filtered, y_filtered)

data_variants = {
    "Original": (X_filtered, y_filtered),
    "SMOTE": (X_sm, y_sm),
    "Tomek": (X_tm, y_tm),
    "SMOTE-Tomek": (X_st, y_st),
    "ADASYN": (X_ad, y_ad)
}

cluster_list = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_noise']
variant_list = ["Original", "SMOTE", "Tomek", "SMOTE-Tomek", "ADASYN"]

s1_result = []
for variant in variant_list:
    #Original
    X = data_variants[variant][0]
    y = data_variants[variant][1]
    df = train_multi_model(variant, X, y)
    s1_result.append(df)

s1_df_final =  pd.concat(s1_result, ignore_index=True)
s1_df_final.to_excel(f'{output_predicting_result}/s1_predict_training_result.xlsx')
#Senario 1 final model
s1_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
s1_X = data_variants['ADASYN'][0]
s1_y = data_variants['ADASYN'][1]
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted'
}
start_t = time.perf_counter()
# Tính toán cross-validation
cv_results = cross_validate(
    s1_model, s1_X, s1_y,
    cv=cv,
    scoring=scoring_metrics,
    error_score='raise',
    n_jobs=-1
)
end_t = time.perf_counter()
total_time = end_t - start_t
r = {
    "Data Variant": "ADASYN",
    "Model": "GradientBoostingClassifier",
    "Accuracy": np.mean(cv_results['test_accuracy']),
    "Std Dev": np.std(cv_results['test_accuracy']),
    "Precision": np.mean(cv_results['test_precision']),
    "Recall": np.mean(cv_results['test_recall']),
    "F1-Score": np.mean(cv_results['test_f1']),
    "Time (s)": np.mean(cv_results['fit_time'])
}
s1_model.fit(s1_X, s1_y)
model_filename = 'GradientBoostingClassifier_model.pkl'
print(f"  + GradientBoostingClassifier training: Hoàn tất (Accuracy: {r['Accuracy']:.4f} | F1: {r['F1-Score']:.4f} | Time : {total_time} )")
joblib.dump(s1_model, os.path.join(output_predicting_model, model_filename))
print(f"Đã lưu mô hình thành công tại: {model_filename}")
#-----------------------------------------------------------------------------------------
#Senario 2: Using each cluster data, not include cluster information to train predict model
from sklearn.model_selection import GridSearchCV

cluster_raw_dict = split_data_by_clusters(final_modeling_df, 'final_label')
final_processed_data = preprocess_clusters(cluster_dict=cluster_raw_dict, target_col='step1')
#Training and Tuning
for cluster in cluster_list:
    if cluster not in final_processed_data:
        continue
    
    result_list = []
    # Find best model for the cluster first using default multi_model train
    for variant in variant_list:
        if variant in final_processed_data[cluster]:
            X, y = final_processed_data[cluster][variant]
            df = train_multi_model(f'{cluster}_{variant}', X, y)
            result_list.append(df)
            
    if result_list:
        df_final = pd.concat(result_list, ignore_index=True)
        df_final.to_excel(f'{output_predicting_result}/s2_{cluster}_predict_training_result.xlsx')
        
        # Improve accuracy: Hyperparameter tuning for the Gradient Boosting model on the ADASYN variant
        if 'ADASYN' in final_processed_data[cluster]:
            print(f"\n--- Tối ưu hóa siêu tham số cho {cluster} ---")
            X_opt, y_opt = final_processed_data[cluster]['ADASYN']
            
            gb_model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            }
            
            grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, 
                                       cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), 
                                       scoring='accuracy', n_jobs=-1)
            
            start_t = time.perf_counter()
            grid_search.fit(X_opt, y_opt)
            end_t = time.perf_counter()
            
            best_model = grid_search.best_estimator_
            print(f"[{cluster}] Best Params: {grid_search.best_params_}")
            print(f"[{cluster}] Best Accuracy: {grid_search.best_score_:.4f} (Time: {end_t - start_t:.2f}s)")
            
            # Save the optimized model for this cluster
            joblib.dump(best_model, os.path.join(output_predicting_model, f's2_gb_optimized_model_{cluster}.pkl'))
