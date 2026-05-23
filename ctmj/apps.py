import os
import joblib
from django.apps import AppConfig
from django.conf import settings
import fsspec

print("Hệ thống Django đang khởi động apps")


class CtmjConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ctmj'
    verbose_name = 'Hệ thống Dự báo CTMJ'
    loaded_resources = {}

    def ready(self):
        BUCKET_ID = "quanghuynh0122/datamining_models"

        # Định nghĩa danh sách file cần nạp (Key là tên bạn muốn gọi ở views, Value là tên file trên Bucket)
        RESOURCES_TO_LOAD = {
            "gradient_boosting": "GradientBoostingClassifier_model.pkl",
            "dbscan": "dbscan_clustering_model.pkl",
            "spectral": "spectral_clustering_model.pkl",
            "predicting_preprocessor": "s1_predicting_preprocessor.pkl",
            "user_data_preprocessor": "user_data_preprocessor.pkl"
        }

        # Kiểm tra luồng chạy chính của Django để tránh bị lặp 2 lần
        if os.environ.get('RUN_MAIN') == 'true' or not settings.DEBUG:

            print("\n" + "=" * 50)
            hf_token = input("HUGGING FACE TOKEN: ").strip()
            print("=" * 50)

            if not hf_token:
                print("Chưa nhập Token!")
                return
            try:
                # Khởi tạo hệ thống file với token vừa nhập
                fs = fsspec.filesystem("hf", token=hf_token)

                for resource_key, file_name in RESOURCES_TO_LOAD.items():
                    bucket_file_path = f"buckets/{BUCKET_ID}/{file_name}"
                    print(f"Đang load {file_name}")

                    with fs.open(bucket_file_path, "rb") as f:

                        CtmjConfig.loaded_resources[resource_key] = joblib.load(f)
                print("Load model thành công")

            except Exception as e:
                print("Có lỗi xảy ra: ", e)