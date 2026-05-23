from ctmj.models import (
    GenderID, BAS_werkzaamheid_resp, SPSS_Regio5, BAS_bruto_jaarinkomen,
    AFG_sk2015, BAS_voltooide_opleiding8_resp, SPSS_Lifestage, type_touch, cluster_info
)


def run():
    print("--- Bắt đầu nạp dữ liệu nhanh vào SQLite ---")

    # 1. GenderID
    genders = [
        {"code": 1, "name": "Male"},
        {"code": 2, "name": "Female"}
    ]
    for item in genders:
        GenderID.objects.update_or_create(gender_code=item["code"], defaults={"gender_name": item["name"]})

    # 2. BAS_werkzaamheid_resp (Tình trạng việc làm)
    jobs = [
        {"code": 1, "name": "Entrepreneur"},
        {"code": 2, "name": "Salaried employment"},
        {"code": 3, "name": "Working for the government"},
        {"code": 4, "name": "Incapacitated"},
        {"code": 5, "name": "Unemployed / Job seeker"},
        {"code": 6, "name": "Social assistance benefit"},
        {"code": 7, "name": "(Early) retirement"},
        {"code": 8, "name": "Student / Scholar"},
        {"code": 9, "name": "Housewife / Houseman / Other"},
        {"code": 97, "name": "Don't know"},
        {"code": 99, "name": "Not applicable"}
    ]
    for item in jobs:
        BAS_werkzaamheid_resp.objects.update_or_create(code=item["code"], defaults={"name": item["name"]})

    # 3. SPSS_Regio5 (Vùng địa lý)
    regions = [
        {"code": 1, "name": "Amsterdam, Rotterdam, Den Haag"},
        {"code": 2, "name": "West (1 excluded)"},
        {"code": 3, "name": "North"},
        {"code": 4, "name": "East"},
        {"code": 5, "name": "South"}
    ]
    for item in regions:
        SPSS_Regio5.objects.update_or_create(code=item["code"], defaults={"name": item["name"]})

    # 7. BAS_bruto_jaarinkomen (Thu nhập năm)
    incomes = [
        {"code": 1, "name": "< €12.900 (minimum)"},
        {"code": 2, "name": "€12.900 <= 27.000 (below average)"},
        {"code": 3, "name": "€27.000 <= 33.500 (almost average)"},
        {"code": 4, "name": "€33.500 <= 40.000 (average)"},
        {"code": 5, "name": "€40.000 <= 67.000 (between 1 and 2 times average)"},
        {"code": 6, "name": "€67.000 <= 79.900 (2 times average)"},
        {"code": 7, "name": "> 79.900 (above 2 times average)"},
        {"code": 8, "name": "Don't know / Don't want to say"}
    ]
    for item in incomes:
        BAS_bruto_jaarinkomen.objects.update_or_create(code=item["code"], defaults={"name": item["name"]})

    # 9. AFG_sk2015 (Tầng lớp xã hội)
    social_classes = [
        {"code": 1, "name": "A"},
        {"code": 2, "name": "B1"},
        {"code": 3, "name": "B2"},
        {"code": 4, "name": "C"},
        {"code": 5, "name": "D"}
    ]
    for item in social_classes:
        AFG_sk2015.objects.update_or_create(code=item["code"], defaults={"name": item["name"]})

    # 10. BAS_voltooide_opleiding8_resp (Trình độ học vấn)
    educations = [
        {"code": 1, "name": "Geen of basisonderwijs"},
        {"code": 2, "name": "LBO / VMBO (kader- of beroepsgericht) / MBO 1 / VBO"},
        {"code": 3, "name": "MAVO / HAVO of VWO (overgegaan naar 4e klas) / VMBO (theoretisch of gemengd) / (M)ULO"},
        {"code": 4, "name": "MBO 2, 3, 4 of MBO voor 1998"},
        {"code": 5, "name": "HAVO of VWO (met diploma afgerond) / HBS / MMS"},
        {"code": 6, "name": "HBO of universitair propedeuse"},
        {"code": 7, "name": "HBO of universitair bachelor / kandidaats"},
        {"code": 8, "name": "HBO of universitair master / doctoraal / postdoctoraal"},
        {"code": 97, "name": "Don't know / Don't want to say"},
        {"code": 99, "name": "Not applicable"}
    ]
    for item in educations:
        BAS_voltooide_opleiding8_resp.objects.update_or_create(code=item["code"], defaults={"name": item["name"]})

    # 11. SPSS_Lifestage (Giai đoạn cuộc sống)
    lifestages = [
        {"code": 1, "name": "Young Singles"},
        {"code": 2, "name": "Mature Singles"},
        {"code": 3, "name": "Young Couples"},
        {"code": 4, "name": "Empty Nesters"},
        {"code": 5, "name": "Young Families"},
        {"code": 6, "name": "Mature Families"},
        {"code": 7, "name": "Established Families"},
        {"code": 8, "name": "Single Parents, child(ren)"},
        {"code": 9, "name": "Single Parents, adult child(ren)"},
        {"code": 97, "name": "Unknown"}
    ]
    for item in lifestages:
        SPSS_Lifestage.objects.update_or_create(code=item["code"], defaults={"name": item["name"]})

    # 12. type_touch (Đã xóa hết dấu cách lỗi ở chữ "description")
    types = [
        {"code": 1, "name": "Accommodations Website",
         "description": "Khách hàng chủ động truy cập trực tiếp vào các trang web cung cấp dịch vụ lưu trú (khách sạn, homestay, resort...) để tìm phòng."},
        {"code": 2, "name": "Accommodations App",
         "description": "Khách hàng chủ động mở và tương tác trên các ứng dụng di động chuyên đặt phòng lưu trú."},
        {"code": 3, "name": "Accommodations Search",
         "description": "Khách hàng chủ động tìm kiếm các từ khóa liên quan đến lưu trú/khách sạn trên các công cụ tìm kiếm (Google, Bing...)."},
        {"code": 4, "name": "Information / Comparison Website",
         "description": "Khách hàng truy cập các website thông tin hoặc so sánh giá (như TripAdvisor, Trivago...) để xem đánh giá và so sánh các lựa chọn."},
        {"code": 5, "name": "Information / Comparison App",
         "description": "Khách hàng sử dụng các ứng dụng di động có chức năng so sánh giá và cung cấp thông tin du lịch/lưu trú/vé máy bay."},
        {"code": 6, "name": "Information / Comparison Search",
         "description": "Khách hàng tìm kiếm trên Google/Bing các từ khóa mang tính chất so sánh, đánh giá."},
        {"code": 7, "name": "Touroperator / Travel agent Website Competitor",
         "description": "Khách hàng truy cập trực tiếp vào trang web của các công ty lữ hành/đại lý du lịch đối thủ cạnh tranh."},
        {"code": 8, "name": "Touroperator / Travel agent App Competitor",
         "description": "Khách hàng sử dụng ứng dụng di động của các công ty lữ hành/đại lý du lịch đối thủ cạnh tranh."},
        {"code": 9, "name": "Touroperator / Travel agent Search Competitor",
         "description": "Khách hàng chủ động tìm kiếm trên Google tên thương hiệu hoặc dịch vụ của các đại lý du lịch đối thủ."},
        {"code": 10, "name": "Touroperator / Travel agent Website Focus brand",
         "description": "Khách hàng truy cập trực tiếp vào trang web của chính thương hiệu mục tiêu."},
        {"code": 12, "name": "Touroperator / Travel agent Search Focus brand",
         "description": "Khách hàng tìm kiếm các từ khóa liên quan trực tiếp đến tên thương hiệu mục tiêu trên các công cụ tìm kiếm."},
        {"code": 13, "name": "Flight tickets Website",
         "description": "Khách hàng chủ động truy cập vào website của các hãng hàng không hoặc đại lý bán vé máy bay."},
        {"code": 14, "name": "Flight tickets App",
         "description": "Khách hàng mở và tương tác trên ứng dụng di động chuyên đặt vé máy bay."},
        {"code": 15, "name": "Flight tickets Search",
         "description": "Khách hàng tìm kiếm các từ khóa liên quan đến vé máy bay, chặng bay trên các công cụ tìm kiếm."},
        {"code": 16, "name": "Generic search",
         "description": "Khách hàng tìm kiếm các từ khóa chung chung, không định hình rõ thương hiệu hay dịch vụ cụ thể."},
        {"code": 18, "name": "Affiliates",
         "description": "Điểm chạm đến từ mạng lưới tiếp thị liên kết. Khách hàng click vào link giới thiệu của các bên thứ ba (KOLs, Blogger, trang mã giảm giá) và được dẫn về doanh nghiệp."},
        {"code": 19, "name": "Banner",
         "description": "Khách hàng nhìn thấy và click vào các banner quảng cáo hiển thị (Display Ads) của doanh nghiệp trên các trang báo điện tử hoặc website đối tác."},
        {"code": 20, "name": "Email",
         "description": "Khách hàng tương tác (mở, click) với các chiến dịch gửi email marketing, bản tin (newsletter) hoặc email khuyến mãi do doanh nghiệp chủ động gửi."},
        {"code": 21, "name": "Prerolls",
         "description": "Khách hàng xem các video quảng cáo ngắn xuất hiện trước khi nội dung video chính bắt đầu (thường gặp trên YouTube hoặc các nền t triển khai video)."},
        {"code": 22, "name": "Retargeting",
         "description": "Quảng cáo bám đuôi (tiếp thị lại). Điểm chạm xảy ra khi hệ thống phân phối quảng cáo lại cho những khách hàng cũ đã từng tương tác với website/app của doanh nghiệp nhằm nhắc nhở họ mua hàng."}
    ]
    for item in types:
        type_touch.objects.update_or_create(
            code=item["code"],
            defaults={"name": item["name"], "description": item["description"]}
        )

    # 13. Clusters (Đồng bộ đồng nhất toàn bộ khóa về tên "id")
    clusters = [
        {
            "id": -1,
            "name": "Noise",
            "desc": "Nhóm khách hàng mang tính chất dị biệt và phân tán cao, không đại diện cho một xu hướng tiêu dùng cụ thể nào trong hành trình mua sắm"
        },
        {
            "id": 0,
            "name": "Empty Nesters & Mature Singles",
            "desc": "Nhóm khách hàng lớn tuổi, đã nghỉ hưu, sống độc thân hoặc hai vợ chồng già với quy mô gia đình nhỏ và không có áp lực nuôi con nhỏ."
        },
        {
            "id": 1,
            "name": "Mature & Established Families",
            "desc": "Nhóm khách hàng trung niên (chủ yếu là phụ nữ) có công việc ổn định, thu nhập khá và đang trong giai đoạn bận rộn chăm sóc gia đình có từ 1 đến 2 con nhỏ"
        },
        {
            "id": 2,
            "name": "Young Singles & Young Couples",
            "desc": "Nhóm khách hàng trẻ tuổi, độc thân hoặc mới lập gia đình, chưa có con, có lối sống năng động và tự do."
        }
    ]
    for item in clusters:
        cluster_info.objects.update_or_create(
            cluster_id=item["id"],
            defaults={
                "name": item["name"],
                "description": item["desc"]
            }
        )

    print("--- Hoàn thành! Đã nạp thành công toàn bộ dữ liệu vào SQLite ---")