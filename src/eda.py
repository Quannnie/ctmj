import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ydata_profiling import ProfileReport
import webbrowser
import os

from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans

#Finetune DBSCAN
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

def data_profiling(df, output_dir, output_filename, dashboard_title):
    output_path = os.path.join(output_dir, output_filename)

    #Check direction
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_cleaned = df.replace('--', np.nan)

    #Data profiling
    print("Running Profile Report...")
    profile = ProfileReport(df_cleaned, title=dashboard_title, minimal=True)
    profile.to_file(output_path)

    #Open profiling
    full_path = os.path.abspath(output_path)
    print(f"Open report: {full_path}")
    webbrowser.open(f"file://{full_path}")

def visualizer_all_metrics(X_scaled):
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    k_range = range(2, 12)

    # 1. Elbow Method (Distortion Score)
    model1 = KMeans(init='k-means++', random_state=42, n_init=10)
    visualizer1 = KElbowVisualizer(model1, k=(2, 12), ax=ax[0], force_model=True)
    visualizer1.fit(X_scaled)
    k_elbow = visualizer1.elbow_value_
    ax[0].set_title(f"Elbow Method (Optimal k={k_elbow})")

    # 2. Silhouette Score
    model2 = KMeans(init='k-means++', random_state=42, n_init=10)
    visualizer2 = KElbowVisualizer(model2, k=(2, 12), metric='silhouette', ax=ax[1], force_model=True)
    visualizer2.fit(X_scaled)
    k_silhouette = visualizer2.elbow_value_
    ax[1].set_title(f"Silhouette Score (Optimal k={k_silhouette})")

    # 3. Davies-Bouldin Index (DBI)
    dbi_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        dbi_scores.append(davies_bouldin_score(X_scaled, labels))

    best_dbi_idx = dbi_scores.index(min(dbi_scores))
    k_dbi = k_range[best_dbi_idx]

    # Vẽ biểu đồ DBI
    ax[2].plot(k_range, dbi_scores, marker='o', color='#1313FF', linewidth=2)
    ax[2].set_xlabel('k')
    ax[2].set_ylabel('DBI Score')
    ax[2].set_title(f'Davies-Bouldin Index (Optimal k={k_dbi})')
    ax[2].axvline(k_dbi, color='black', linestyle='--', label=f'Best k = {k_dbi}')
    ax[2].grid(True, alpha=0.3)
    ax[2].legend()

    plt.tight_layout()
    plt.show()

def find_optimal_eps(data, k):
    #Tính khoảng cách K-neighbors
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(data)
    distances, _ = nbrs.kneighbors(data)

    # Lấy khoảng cách láng giềng xa nhất và sắp xếp
    sorted_distances = np.sort(distances[:, k-1], axis=0)
    x_values = np.arange(len(sorted_distances))

    #Sử dụng KneeLocator để tìm điểm gãy
    #curve='convex' vì đường cong lõm lên trên, direction='increasing'
    kneedle = KneeLocator(x_values, sorted_distances,
                          curve='convex',
                          direction='increasing',
                          interp_method='polynomial')

    eps_opt = sorted_distances[kneedle.knee]

    #Vẽ đồ thị kết quả
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances, label='K-distance')
    plt.axhline(y=eps_opt, color='r', linestyle='--', label=f'Optimal Eps: {eps_opt:.4f}')
    plt.axvline(x=kneedle.knee, color='g', linestyle='--', label=f'Knee Point: {kneedle.knee}')

    plt.title(f'Elbow Detection (k={k})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return eps_opt
