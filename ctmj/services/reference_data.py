"""Authoritative reference data for the CJPS lookup tables.

This module is the single source of truth for every dropdown in the UI. It is
consumed by:

  * ``ctmj/migrations/0010_seed_reference_data.py`` — populates a fresh
    database so ``migrate`` alone is enough to get a working install.
  * ``manage.py seed_reference_data`` — idempotent re-sync for existing
    databases.

Keeping the data here (rather than inside a migration) means it can be updated
and re-applied, and it can be imported without touching the migration graph.
"""

from __future__ import annotations

from typing import Any

# Each table is described once: model name, primary-key field, and rows.
# Field names match the Django models in ``ctmj.models``.

GENDERS: list[dict[str, Any]] = [
    {"gender_code": 1, "gender_name": "Male"},
    {"gender_code": 2, "gender_name": "Female"},
]

# Tình trạng việc làm (employment status)
JOBS: list[dict[str, Any]] = [
    {"code": 1, "name": "Entrepreneur"},
    {"code": 2, "name": "Salaried employment"},
    {"code": 3, "name": "Working for the government"},
    {"code": 4, "name": "Incapacitated"},
    {"code": 5, "name": "Unemployed / job seeker"},
    {"code": 6, "name": "Social assistance benefit"},
    {"code": 7, "name": "(Early) retirement"},
    {"code": 8, "name": "Student / scholar"},
    {"code": 9, "name": "Housewife / houseman / other"},
    {"code": 97, "name": "Don't know"},
    {"code": 99, "name": "Not applicable"},
]

# Vùng địa lý (CBS/Netherlands regional classification, 5 groups)
REGIONS: list[dict[str, Any]] = [
    {"code": 1, "name": "Amsterdam, Rotterdam, Den Haag"},
    {"code": 2, "name": "West (Randstad excluded)"},
    {"code": 3, "name": "North"},
    {"code": 4, "name": "East"},
    {"code": 5, "name": "South"},
]

# Tổng thu nhập hộ (gross annual household income, EUR bands)
INCOMES: list[dict[str, Any]] = [
    {"code": 1, "name": "Under 12,900 (minimum)"},
    {"code": 2, "name": "12,900 to 27,000 (below average)"},
    {"code": 3, "name": "27,000 to 33,500 (almost average)"},
    {"code": 4, "name": "33,500 to 40,000 (average)"},
    {"code": 5, "name": "40,000 to 67,000 (1 to 2x average)"},
    {"code": 6, "name": "67,000 to 79,900 (2x average)"},
    {"code": 7, "name": "Over 79,900 (above 2x average)"},
    {"code": 8, "name": "Don't know / don't want to say"},
]

# Tầng lớp xã hội (CBS social class, 2015 standard)
SOCIAL_CLASSES: list[dict[str, Any]] = [
    {"code": 1, "name": "A — high status"},
    {"code": 2, "name": "B1 — above average"},
    {"code": 3, "name": "B2 — average"},
    {"code": 4, "name": "C — below average"},
    {"code": 5, "name": "D — low status"},
]

# Trình độ học vấn (highest completed level)
EDUCATIONS: list[dict[str, Any]] = [
    {"code": 1, "name": "No or primary education"},
    {"code": 2, "name": "Lower secondary / vocational / MBO level 1"},
    {"code": 3, "name": "Upper secondary (MAVO, HAVO, VWO)"},
    {"code": 4, "name": "MBO levels 2, 3, 4"},
    {"code": 5, "name": "Senior secondary (HAVO, VWO with diploma)"},
    {"code": 6, "name": "Higher education, propedeuse"},
    {"code": 7, "name": "Higher education, bachelor"},
    {"code": 8, "name": "Higher education, master or doctorate"},
    {"code": 97, "name": "Don't know / don't want to say"},
    {"code": 99, "name": "Not applicable"},
]

# Giai đoạn cuộc sống (life-stage segment)
LIFESTAGES: list[dict[str, Any]] = [
    {"code": 1, "name": "Young singles"},
    {"code": 2, "name": "Mature singles"},
    {"code": 3, "name": "Young couples"},
    {"code": 4, "name": "Empty nesters"},
    {"code": 5, "name": "Young families"},
    {"code": 6, "name": "Mature families"},
    {"code": 7, "name": "Established families"},
    {"code": 8, "name": "Single parents, child(ren)"},
    {"code": 9, "name": "Single parents, adult child(ren)"},
    {"code": 97, "name": "Unknown"},
]

# Điểm chạm / kênh tiếp thị (customer-journey touchpoints)
TOUCHPOINTS: list[dict[str, Any]] = [
    {
        "code": 1,
        "name": "Accommodation website",
        "description": (
            "Khách hàng chủ động truy cập trực tiếp vào website cung cấp dịch vụ "
            "lưu trú (khách sạn, homestay, resort) để tìm phòng."
        ),
    },
    {
        "code": 2,
        "name": "Accommodation app",
        "description": (
            "Khách hàng chủ động mở và tương tác trên ứng dụng di động chuyên "
            "đặt phòng lưu trú."
        ),
    },
    {
        "code": 3,
        "name": "Accommodation search",
        "description": (
            "Khách hàng chủ động tìm kiếm từ khóa liên quan đến lưu trú trên công cụ "
            "tìm kiếm (Google, Bing)."
        ),
    },
    {
        "code": 4,
        "name": "Information / comparison website",
        "description": (
            "Khách hàng truy cập website thông tin hoặc so sánh giá (TripAdvisor, "
            "Trivago) để xem đánh giá và so sánh lựa chọn."
        ),
    },
    {
        "code": 5,
        "name": "Information / comparison app",
        "description": (
            "Khách hàng sử dụng ứng dụng di động có chức năng so sánh giá và cung "
            "cấp thông tin du lịch, lưu trú, vé máy bay."
        ),
    },
    {
        "code": 6,
        "name": "Information / comparison search",
        "description": (
            "Khách hàng tìm kiếm trên Google hoặc Bing các từ khóa mang tính so "
            "sánh, đánh giá."
        ),
    },
    {
        "code": 7,
        "name": "Competitor website",
        "description": (
            "Khách hàng truy cập trực tiếp vào website của công ty lữ hành hoặc "
            "đại lý du lịch đối thủ cạnh tranh."
        ),
    },
    {
        "code": 8,
        "name": "Competitor app",
        "description": (
            "Khách hàng sử dụng ứng dụng di động của công ty lữ hành hoặc đại lý "
            "du lịch đối thủ cạnh tranh."
        ),
    },
    {
        "code": 9,
        "name": "Competitor search",
        "description": (
            "Khách hàng chủ động tìm tên thương hiệu hoặc dịch vụ của đại lý du "
            "lịch đối thủ trên Google."
        ),
    },
    {
        "code": 10,
        "name": "Focus brand website",
        "description": "Khách hàng truy cập trực tiếp vào website của thương hiệu mục tiêu.",
    },
    {
        "code": 12,
        "name": "Focus brand search",
        "description": (
            "Khách hàng tìm kiếm từ khóa trực tiếp liên quan đến tên thương hiệu "
            "mục tiêu trên công cụ tìm kiếm."
        ),
    },
    {
        "code": 13,
        "name": "Flight ticket website",
        "description": (
            "Khách hàng chủ động truy cập vào website của hãng hàng không hoặc đại lý "
            "bán vé máy bay."
        ),
    },
    {
        "code": 14,
        "name": "Flight ticket app",
        "description": "Khách hàng mở và tương tác trên ứng dụng di động chuyên đặt vé máy bay.",
    },
    {
        "code": 15,
        "name": "Flight ticket search",
        "description": (
            "Khách hàng tìm kiếm từ khóa liên quan đến vé máy bay, chặng bay trên "
            "công cụ tìm kiếm."
        ),
    },
    {
        "code": 16,
        "name": "Generic search",
        "description": (
            "Khách hàng tìm kiếm từ khóa chung chung, không định hình rõ thương hiệu "
            "hay dịch vụ cụ thể."
        ),
    },
    {
        "code": 18,
        "name": "Affiliate",
        "description": (
            "Điểm chạm đến từ mạng lưới tiếp thị liên kết. Khách hàng nhấp vào link "
            "giới thiệu của bên thứ ba (KOL, blogger, trang mã giảm giá) và được "
            "dẫn về doanh nghiệp."
        ),
    },
    {
        "code": 19,
        "name": "Display banner",
        "description": (
            "Khách hàng nhìn thấy và nhấp vào banner quảng cáo hiển thị trên báo điện "
            "tử hoặc website đối tác."
        ),
    },
    {
        "code": 20,
        "name": "Email",
        "description": (
            "Khách hàng tương tác với chiến dịch email marketing, bản tin hoặc email "
            "khuyến mãi do doanh nghiệp chủ động gửi."
        ),
    },
    {
        "code": 21,
        "name": "Video preroll",
        "description": (
            "Khách hàng xem video quảng cáo ngắn xuất hiện trước nội dung chính, phổ "
            "biến trên YouTube và các nền tảng video."
        ),
    },
    {
        "code": 22,
        "name": "Retargeting",
        "description": (
            "Quảng cáo bám đuôi. Điểm chạm phát sinh khi hệ thống phân phối lại quảng "
            "cáo cho khách hàng cũ từng tương tác với website hoặc app của doanh nghiệp."
        ),
    },
]

# Nhóm khách hàng sau phân cụm (cluster_id, name, description)
CLUSTERS: list[dict[str, Any]] = [
    {
        "cluster_id": -1,
        "name": "Noise",
        "description": (
            "Nhóm khách hàng mang tính dị biệt và phân tán cao, không đại diện cho một "
            "xu hướng tiêu dùng cụ thể nào trong hành trình mua sắm."
        ),
    },
    {
        "cluster_id": 0,
        "name": "Empty nesters & mature singles",
        "description": (
            "Khách hàng lớn tuổi, đã nghỉ hưu, sống độc thân hoặc hai vợ chồng già với "
            "quy mô gia đình nhỏ và không có áp lực nuôi con nhỏ."
        ),
    },
    {
        "cluster_id": 1,
        "name": "Mature & established families",
        "description": (
            "Khách hàng trung niên, chủ yếu là phụ nữ, có công việc ổn định, thu nhập khá "
            "và đang trong giai đoạn bận rộn chăm sóc gia đình có từ 1 đến 2 con nhỏ."
        ),
    },
    {
        "cluster_id": 2,
        "name": "Young singles & young couples",
        "description": (
            "Khách hàng trẻ tuổi, độc thân hoặc mới lập gia đình, chưa có con, có lối "
            "sống năng động và tự do."
        ),
    },
]

#: Human labels used for each cluster, keyed by id. -1 is a special case: it is
#: not "no data", it is a real DBSCAN outcome.
CLUSTER_LABELS: dict[int, str] = {
    -1: "Nhiễu",
    0: "Trống tổ và độc thân",
    1: "Gia đình trưởng thành",
    2: "Độc thân và cặp đôi trẻ",
}

#: (model class name, primary-key field, rows, ordering) in dependency order.
REFERENCE_TABLES: tuple[tuple[str, str, list[dict[str, Any]]], ...] = (
    ("GenderID", "gender_code", GENDERS),
    ("BAS_werkzaamheid_resp", "code", JOBS),
    ("SPSS_Regio5", "code", REGIONS),
    ("BAS_bruto_jaarinkomen", "code", INCOMES),
    ("AFG_sk2015", "code", SOCIAL_CLASSES),
    ("BAS_voltooide_opleiding8_resp", "code", EDUCATIONS),
    ("SPSS_Lifestage", "code", LIFESTAGES),
    ("type_touch", "code", TOUCHPOINTS),
    ("cluster_info", "cluster_id", CLUSTERS),
)
