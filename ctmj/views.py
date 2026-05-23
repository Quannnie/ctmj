from django.shortcuts import render
import joblib
import pandas as pd
import os
from .models import (
    GenderID, BAS_werkzaamheid_resp, SPSS_Regio5,
    BAS_bruto_jaarinkomen, AFG_sk2015, BAS_voltooide_opleiding8_resp,
    SPSS_Lifestage, type_touch, cluster_info  # 🌟 Đã import thêm cluster_info ở đây
)
from .apps import CtmjConfig
import numpy as np
from scipy.spatial.distance import cdist


def home_view(request):
    return render(request, 'home.html')


def predict_view(request):
    result = None
    error_message = None
    form_data = {}

    # Lấy model và processor từ RAM (AppConfig)
    user_processor = CtmjConfig.loaded_resources.get('user_data_preprocessor')
    predict_processor = CtmjConfig.loaded_resources.get('predicting_preprocessor')

    model_dbscan = CtmjConfig.loaded_resources.get('dbscan')
    model_spectral = CtmjConfig.loaded_resources.get('spectral')
    model_gb = CtmjConfig.loaded_resources.get('gradient_boosting')

    if request.method == 'POST':
        try:
            # 1. Thu thập và kiểm tra dữ liệu từ Form gửi lên
            required_fields = [
                'GenderID', 'Age', 'SPSS_Regio5', 'BAS_huishoudgrootte',
                'BAS_werkzaamheid_resp', 'BAS_bruto_jaarinkomen',
                'afg_kinderen_huishouden', 'AFG_sk2015',
                'BAS_voltooide_opleiding8_resp', 'SPSS_Lifestage',
                'step_1_channel', 'step_2_channel'
            ]

            # Check điền thiếu thông tin
            for field in required_fields:
                val = request.POST.get(field)
                if not val or val.strip() == "":
                    raise ValueError(f"Thiếu thông tin cho trường: {field}")
                form_data[field] = val

            # Validate các trường dữ liệu dạng số tự do nhập
            try:
                age = int(form_data['Age'])
                kids = int(form_data['afg_kinderen_huishouden'])
                h_grootte = int(form_data['BAS_huishoudgrootte'])

                if age < 0 or age > 120:
                    raise ValueError("Tuổi phải nằm trong khoảng từ 0 đến 120.")
                if kids < 0 or h_grootte <= 0:
                    raise ValueError("Số trẻ em hoặc quy mô hộ gia đình không hợp lệ.")
            except ValueError as ve:
                if "invalid literal" in str(ve):
                    raise ValueError("Tuổi, Số trẻ em và Quy mô gia đình phải là số nguyên.")
                else:
                    raise ve

            # 2. XÂY DỰNG DATAFRAME ĐẦU VÀO
            raw_input_data = pd.DataFrame([{
                'BAS_huishoudgrootte': h_grootte,
                'BAS_werkzaamheid_resp': int(form_data['BAS_werkzaamheid_resp']),
                'afg_kinderen_huishouden': kids,
                'BAS_voltooide_opleiding8_resp': int(form_data['BAS_voltooide_opleiding8_resp']),
                'SPSS_Lifestage': int(form_data['SPSS_Lifestage']),
                'GenderID': int(form_data['GenderID']),
                'SPSS_Regio5': int(form_data['SPSS_Regio5']),
                'BAS_bruto_jaarinkomen': int(form_data['BAS_bruto_jaarinkomen']),
                'AFG_sk2015': int(form_data['AFG_sk2015']),
                'Age': age,
                'step2': int(form_data['step_1_channel']),
                'step3': int(form_data['step_2_channel'])
            }])

            # --- BƯỚC I: XỬ LÝ PHÂN CỤM LAI ---
            clustering_feature = [
                'BAS_huishoudgrootte', 'BAS_werkzaamheid_resp', 'afg_kinderen_huishouden',
                'BAS_voltooide_opleiding8_resp', 'SPSS_Lifestage', 'GenderID',
                'SPSS_Regio5', 'BAS_bruto_jaarinkomen', 'AFG_sk2015', 'Age'
            ]
            df_cluster = raw_input_data[clustering_feature]

            if user_processor is None or model_dbscan is None or model_spectral is None:
                raise ValueError("Bộ tiền xử lý hoặc các mô hình Phân cụm (DBSCAN/Spectral) chưa được nạp.")

            X_cluster_scaled = user_processor.transform(df_cluster)

            # 1. KIỂM TRA ĐIỀU KIỆN NGOẠI LAI BẰNG DBSCAN
            if hasattr(model_dbscan, 'components_') and model_dbscan.components_.shape[0] > 0:
                dbscan_dists = cdist(X_cluster_scaled, model_dbscan.components_, metric='euclidean')
                closest_core_idx = np.argmin(dbscan_dists[0])
                dbscan_temp_label = int(model_dbscan.labels_[model_dbscan.core_sample_indices_[closest_core_idx]])
            else:
                dbscan_temp_label = -1

            # 2. RẼ NHÁNH LOGIC THEO YÊU CẦU
            if dbscan_temp_label == -1:
                final_label_val = -1
            else:
                if hasattr(model_spectral, 'components_'):
                    spectral_dists = cdist(X_cluster_scaled, model_spectral.components_, metric='euclidean')
                    closest_spec_idx = np.argmin(spectral_dists[0])
                    final_label_val = int(model_spectral.labels_[closest_spec_idx])
                elif hasattr(model_dbscan, 'components_'):
                    spectral_dists = cdist(X_cluster_scaled, model_dbscan.components_, metric='euclidean')
                    closest_idx_in_train = model_dbscan.core_sample_indices_[np.argmin(spectral_dists[0])]
                    final_label_val = int(model_spectral.labels_[closest_idx_in_train])
                else:
                    final_label_val = dbscan_temp_label

            raw_input_data['final_label'] = final_label_val

            # --- BƯỚC II: XỬ LÝ DỰ BÁO TOP 3 KÊNH TIẾP XÚC (GRADIENT BOOSTING) ---
            predict_features = [
                'BAS_werkzaamheid_resp', 'afg_kinderen_huishouden', 'GenderID',
                'SPSS_Regio5', 'final_label', 'step2', 'step3', 'Age'
            ]
            df_predict = raw_input_data[predict_features]

            if predict_processor is None or model_gb is None:
                raise ValueError("Bộ dự báo hoặc mô hình Gradient Boosting chưa được nạp vào RAM.")

            if hasattr(model_gb, 'best_estimator_'):
                estimator = model_gb.best_estimator_
            else:
                estimator = model_gb

            X_predict_scaled = predict_processor.transform(df_predict)

            probabilities = estimator.predict_proba(X_predict_scaled)[0]
            class_labels = estimator.classes_

            prob_list = []
            for label, prob in zip(class_labels, probabilities):
                prob_list.append({
                    'channel_id': int(label),
                    'probability': float(prob) * 100
                })

            prob_list = sorted(prob_list, key=lambda x: x['probability'], reverse=True)
            top_3_channels = prob_list[:3]

            # =======================================================
            # 🚀 ĐOẠN ĐƯỢC THÊM MỚI: TRUY VẤN ĐỘNG DATABASE CHO RESULT
            # ========================================================

            # 1. Khởi tạo cục dữ liệu gốc ban đầu
            result = {
                'status': 'success',
                'cluster': final_label_val,
                'top_3_probabilities': top_3_channels
            }

            # 2. Lấy thông tin Cluster thực tế (Tên cụm và mô tả tiếng Việt) từ bảng cluster_info
            try:
                cluster_obj = cluster_info.objects.get(cluster_id=final_label_val)
                result['cluster_name'] = cluster_obj.name
                result['cluster_description'] = cluster_obj.description
            except cluster_info.DoesNotExist:
                result['cluster_name'] = f"Nhóm phân cụm đặc trưng #{final_label_val}"
                result[
                    'cluster_description'] = "Hệ thống ghi nhận hành vi thuộc nhóm đặc trưng số này từ tệp huấn luyện."

            # 3. Lấy thông tin Tên kênh (Name) và Mô tả chi tiết (Description) từ bảng type_touch
            for item in result['top_3_probabilities']:
                try:
                    touch_obj = type_touch.objects.get(code=item['channel_id'])
                    item['channel_name'] = touch_obj.name
                    item['channel_description'] = touch_obj.description
                except type_touch.DoesNotExist:
                    item['channel_name'] = "Kênh hành trình mới"
                    item[
                        'channel_description'] = "Hệ thống khuyến nghị tiếp cận và thiết lập chiến dịch qua kênh tương tác này."

            print(result)
        except ValueError as e:
            error_message = str(e)
        except Exception as e:
            error_message = f"Đã xảy ra lỗi hệ thống khi dự báo: {str(e)}"

    # 3. Truy vấn dữ liệu từ SQLite đưa vào context hiển thị Form Dropdown
    context = {
        'genders': GenderID.objects.all(),
        'jobs': BAS_werkzaamheid_resp.objects.all(),
        'regions': SPSS_Regio5.objects.all(),
        'incomes': BAS_bruto_jaarinkomen.objects.all(),
        'social_classes': AFG_sk2015.objects.all(),
        'educations': BAS_voltooide_opleiding8_resp.objects.all(),
        'lifestages': SPSS_Lifestage.objects.all(),
        'touchpoints': type_touch.objects.all(),
        'result': result,
        'error_message': error_message,
        'saved_data': form_data
    }

    return render(request, 'predict.html', context)