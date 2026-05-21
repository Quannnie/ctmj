from ctmj.models import (
GenderID, BAS_werkzaamheid_resp, SPSS_Regio5, BAS_bruto_jaarinkomen,
 AFG_sk2015, BAS_voltooide_opleiding8_resp, SPSS_Lifestage, type_touch
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
    # type touch
    types = [
    {"code": 1, "name": "Accommodations Website"},
    {"code": 2, "name": "Accommodations App"},
    {"code": 3, "name": "Accommodations Search"},
    {"code": 4, "name": "Information / Comparison Website"},
    {"code": 5, "name": "Information / Comparison App"},
    {"code": 6, "name": "Information / Comparison Search"},
    {"code": 7, "name": "Touroperator / Travel agent Website Competitor"},
    {"code": 8, "name": "Touroperator / Travel agent App Competitor"},
    {"code": 9, "name": "Touroperator / Travel agent Search Competitor"},
    {"code": 10, "name": "Touroperator / Travel agent Website Focus brand"},
    {"code": 12, "name": "Touroperator / Travel agent Search Focus brand"},
    {"code": 13, "name": "Flight tickets Website"},
    {"code": 14, "name": "Flight tickets App"},
    {"code": 15, "name": "Flight tickets Search"},
    {"code": 16, "name": "Generic search"},
    {"code": 18, "name": "Affiliates"},
    {"code": 19, "name": "Banner"},
    {"code": 20, "name": "Email"},
    {"code": 21, "name": "Prerolls"},
    {"code": 22, "name": "Retargeting"}
]
    for item in types:
        type_touch.objects.update_or_create(
            code=item["code"],
            defaults={"name": item["name"]}
        )

    print("--- Hoàn thành! Đã nạp thành công toàn bộ dữ liệu vào SQLite ---")