from django.shortcuts import render
import joblib  # hoặc pickle để load model của bạn
import os
from .models import (
    GenderID, BAS_werkzaamheid_resp, SPSS_Regio5,
    BAS_bruto_jaarinkomen, AFG_sk2015, BAS_voltooide_opleiding8_resp,
    SPSS_Lifestage, type_touch
)

# Đường dẫn tới file model của bạn
MODEL_PATH = os.path.join('resources', 'model', 'clustering', 'model', 'dbscan_clustering_model.pkl')

from django.shortcuts import render

def home_view(request):
    return render(request, 'home.html')

def predict_view(request):
    result = None
    error_message = None
    
    if request.method == 'POST':
        try:
            # 1. Thu thập và kiểm tra dữ liệu từ Form
            required_fields = [
                'GenderID', 'Age', 'SPSS_Regio5', 'BAS_huishoudgrootte', 
                'BAS_werkzaamheid_resp', 'BAS_bruto_jaarinkomen', 
                'afg_kinderen_huishouden', 'AFG_sk2015', 
                'BAS_voltooide_opleiding8_resp', 'SPSS_Lifestage', 
                'step_1_channel', 'step_2_channel'
            ]
            
            # Check for missing fields
            form_data = {}
            for field in required_fields:
                val = request.POST.get(field)
                if not val or val.strip() == "":
                    raise ValueError(f"Thiếu thông tin cho trường: {field}")
                form_data[field] = val
                
            # Validate numerical fields
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

            # 2. Xử lý Load Model an toàn
            try:
                # model = joblib.load(MODEL_PATH)
                # X_new = [...] # Tiền xử lý dữ liệu ở đây
                # prediction = model.predict(X_new)
                pass
            except Exception as e:
                # Nếu không tìm thấy file model hoặc lỗi khi load, ghi log hoặc xử lý an toàn
                print(f"Warning: Model loading failed: {e}")
                
            # Trả về kết quả (hiện tại hardcode cho UI)
            result = {
                'status': 'success',
                'cluster': 3,
            }
            
        except ValueError as e:
            error_message = str(e)
        except Exception as e:
            error_message = f"Đã xảy ra lỗi hệ thống: {str(e)}"

    # 1. Truy vấn dữ liệu từ SQLite (Tối ưu hóa có thể dùng select_related/prefetch_related nếu có quan hệ)
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
        'error_message': error_message
    }

    return render(request, 'predict.html', context)