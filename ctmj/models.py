"""Database models.

All tables are reference/lookup data. They are populated by the data migration
``0011_seed_reference_data`` from :mod:`ctmj.services.reference_data`, so a
fresh clone needs nothing more than ``migrate``.

Model names mirror the CBS/Netherlands survey variable names used throughout the
original research code, which keeps the mapping to the trained pipeline obvious.
"""

from __future__ import annotations

from django.contrib import admin
from django.db import models


class GenderID(models.Model):
    gender_code = models.IntegerField(primary_key=True, verbose_name="Mã giới tính")
    gender_name = models.CharField(max_length=20, verbose_name="Tên giới tính")

    class Meta:
        verbose_name = "1. Danh mục Giới tính"
        verbose_name_plural = "1. Danh mục Giới tính"
        ordering = ["gender_code"]

    def __str__(self) -> str:
        return f"{self.gender_code}: {self.gender_name}"


class BAS_werkzaamheid_resp(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã trạng thái")
    name = models.CharField(max_length=255, verbose_name="Tình trạng việc làm (Employment)")

    class Meta:
        verbose_name = "2. Tình trạng việc làm (BAS_werkzaamheid_resp)"
        verbose_name_plural = "2. Tình trạng việc làm (BAS_werkzaamheid_resp)"
        ordering = ["code"]

    def __str__(self) -> str:
        return f"{self.code}: {self.name}"


class SPSS_Regio5(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã vùng")
    name = models.CharField(max_length=255, verbose_name="Vùng địa lý (Region)")

    class Meta:
        verbose_name = "3. Vùng địa lý (SPSS_Regio5)"
        verbose_name_plural = "3. Vùng địa lý (SPSS_Regio5)"
        ordering = ["code"]

    def __str__(self) -> str:
        return f"{self.code}: {self.name}"


class BAS_bruto_jaarinkomen(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã thu nhập")
    name = models.CharField(max_length=255, verbose_name="Tổng thu nhập năm (Gross annual income)")

    class Meta:
        verbose_name = "7. Thu nhập năm (BAS_bruto_jaarinkomen)"
        verbose_name_plural = "7. Thu nhập năm (BAS_bruto_jaarinkomen)"
        ordering = ["code"]

    def __str__(self) -> str:
        return f"{self.code}: {self.name}"


class AFG_sk2015(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã tầng lớp")
    name = models.CharField(max_length=255, verbose_name="Tầng lớp xã hội (Social class 2015)")

    class Meta:
        verbose_name = "9. Tầng lớp xã hội (AFG_sk2015)"
        verbose_name_plural = "9. Tầng lớp xã hội (AFG_sk2015)"
        ordering = ["code"]

    def __str__(self) -> str:
        return f"{self.code}: {self.name}"


class BAS_voltooide_opleiding8_resp(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã học vấn")
    name = models.CharField(max_length=255, verbose_name="Trình độ học vấn (Education level)")

    class Meta:
        verbose_name = "10. Trình độ học vấn (BAS_voltooide_opleiding8_resp)"
        verbose_name_plural = "10. Trình độ học vấn (BAS_voltooide_opleiding8_resp)"
        ordering = ["code"]

    def __str__(self) -> str:
        return f"{self.code}: {self.name}"


class SPSS_Lifestage(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã giai đoạn")
    name = models.CharField(max_length=255, verbose_name="Giai đoạn cuộc sống (Lifestage)")

    class Meta:
        verbose_name = "11. Giai đoạn cuộc sống (SPSS_Lifestage)"
        verbose_name_plural = "11. Giai đoạn cuộc sống (SPSS_Lifestage)"
        ordering = ["code"]

    def __str__(self) -> str:
        return f"{self.code}: {self.name}"


class type_touch(models.Model):
    """Customer-journey touchpoints — the prediction target vocabulary."""

    code = models.IntegerField(primary_key=True, verbose_name="Mã touch point")
    name = models.CharField(max_length=255, verbose_name="Tên touch point")
    description = models.TextField(null=True, blank=True)

    class Meta:
        verbose_name = "12. Touch point"
        verbose_name_plural = "12. Touch point"
        ordering = ["code"]

    def __str__(self) -> str:
        return f"{self.code}: {self.name}"


class cluster_info(models.Model):
    """Segment produced by the clustering stage.

    ``cluster_id = -1`` is not a missing value: it is DBSCAN's noise label and a
    real, reportable outcome.
    """

    cluster_id = models.IntegerField(unique=True, primary_key=True)
    name = models.CharField(max_length=255, null=True, blank=True, verbose_name="Tên cụm")
    description = models.TextField(verbose_name="Mô tả cụm")

    class Meta:
        verbose_name = "Cluster information"
        verbose_name_plural = "Cluster information"
        ordering = ["cluster_id"]

    def __str__(self) -> str:
        return f"Cluster #{self.cluster_id}: {self.name or '—'}"


class PredictionRun(models.Model):
    """Audit trail of prediction requests.

    Append-only, and genuinely useful: it records which inputs produced which
    segment, which is what makes a marketing recommendation auditable.
    """

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    cluster_id = models.IntegerField(verbose_name="Mã cụm")
    top_channel = models.IntegerField(verbose_name="Kênh xếp hạng 1")
    top_probability = models.FloatField(verbose_name="Xác suất (%)")
    age = models.PositiveSmallIntegerField(verbose_name="Tuổi")
    household_size = models.PositiveSmallIntegerField(verbose_name="Quy mô hộ")
    children = models.PositiveSmallIntegerField(verbose_name="Số trẻ em")
    step_previous = models.IntegerField(verbose_name="Kênh trước")
    step_last = models.IntegerField(verbose_name="Kênh cuối")

    class Meta:
        verbose_name = "Lịch sử dự báo"
        verbose_name_plural = "Lịch sử dự báo"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"#{self.pk} cluster={self.cluster_id} p={self.top_probability:.1f}%"


class _ReferenceTableAdmin(admin.ModelAdmin):
    """Shared admin configuration for the code/name lookup tables."""

    search_fields = ("name",)
    list_per_page = 50


@admin.register(GenderID)
class GenderIDAdmin(_ReferenceTableAdmin):
    list_display = ("gender_code", "gender_name")


@admin.register(BAS_werkzaamheid_resp)
@admin.register(SPSS_Regio5)
@admin.register(BAS_bruto_jaarinkomen)
@admin.register(AFG_sk2015)
@admin.register(BAS_voltooide_opleiding8_resp)
@admin.register(SPSS_Lifestage)
@admin.register(type_touch)
class _CodeNameAdmin(_ReferenceTableAdmin):
    list_display = ("code", "name")


@admin.register(cluster_info)
class ClusterInfoAdmin(admin.ModelAdmin):
    list_display = ("cluster_id", "name")
    search_fields = ("name", "description")


@admin.register(PredictionRun)
class PredictionRunAdmin(admin.ModelAdmin):
    list_display = ("created_at", "cluster_id", "top_channel", "top_probability", "age")
    list_filter = ("cluster_id",)

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False
