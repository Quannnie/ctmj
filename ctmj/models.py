from django.db import models
from django.contrib import admin

class GenderID(models.Model):
    gender_code = models.IntegerField(primary_key=True, verbose_name="Mã giới tính")
    gender_name = models.CharField(max_length=20, verbose_name="Tên giới tính")

    def __str__(self):
        return f"{self.gender_code}: {self.gender_name}"

    class Meta:
        verbose_name = "1. Danh mục Giới tính"
        verbose_name_plural = "1. Danh mục Giới tính"

class BAS_werkzaamheid_resp(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã trạng thái")
    name = models.CharField(max_length=255, verbose_name="Tình trạng việc làm (Employment)")

    def __str__(self):
        return f"{self.code}: {self.name}"

    class Meta:
        verbose_name = "2. Tình trạng việc làm (BAS_werkzaamheid_resp)"
        verbose_name_plural = "2. Tình trạng việc làm (BAS_werkzaamheid_resp)"

class SPSS_Regio5(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã vùng")
    name = models.CharField(max_length=255, verbose_name="Vùng địa lý (Region)")

    def __str__(self):
        return f"{self.code}: {self.name}"

    class Meta:
        verbose_name = "3. Vùng địa lý (SPSS_Regio5)"
        verbose_name_plural = "3. Vùng địa lý (SPSS_Regio5)"


class BAS_bruto_jaarinkomen(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã thu nhập")
    name = models.CharField(max_length=255, verbose_name="Tổng thu nhập năm (Gross annual income)")

    def __str__(self):
        return f"{self.code}: {self.name}"

    class Meta:
        verbose_name = "7. Thu nhập năm (BAS_bruto_jaarinkomen)"
        verbose_name_plural = "7. Thu nhập năm (BAS_bruto_jaarinkomen)"


class AFG_sk2015(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã tầng lớp")
    name = models.CharField(max_length=255, verbose_name="Tầng lớp xã hội (Social class 2015)")

    def __str__(self):
        return f"{self.code}: {self.name}"

    class Meta:
        verbose_name = "9. Tầng lớp xã hội (AFG_sk2015)"
        verbose_name_plural = "9. Tầng lớp xã hội (AFG_sk2015)"


class BAS_voltooide_opleiding8_resp(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã học vấn")
    name = models.CharField(max_length=255, verbose_name="Trình độ học vấn (Education level)")

    def __str__(self):
        return f"{self.code}: {self.name}"

    class Meta:
        verbose_name = "10. Trình độ học vấn (BAS_voltooide_opleiding8_resp)"
        verbose_name_plural = "10. Trình độ học vấn (BAS_voltooide_opleiding8_resp)"


class SPSS_Lifestage(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã giai đoạn")
    name = models.CharField(max_length=255, verbose_name="Giai đoạn cuộc sống (Lifestage)")

    def __str__(self):
        return f"{self.code}: {self.name}"

    class Meta:
        verbose_name = "11. Giai đoạn cuộc sống (SPSS_Lifestage)"
        verbose_name_plural = "11. Giai đoạn cuộc sống (SPSS_Lifestage)"

class type_touch(models.Model):
    code = models.IntegerField(primary_key=True, verbose_name="Mã touch point")
    name = models.CharField(max_length=255, verbose_name="Tên touch point")
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.code}: {self.name}"

    class Meta:
        verbose_name = "12. Touch point"
        verbose_name_plural = "12. Touch point"

class cluster_info(models.Model):
    cluster_id = models.IntegerField(unique=True, primary_key=True)
    name = models.CharField(max_length=255, verbose_name="Tên cụm", null=True, blank=True) # 🌟 Thêm dòng này
    description = models.TextField(verbose_name="Mô tả cụm")

    def __str__(self):
        return f"Cluster #{self.cluster_id}: {self.name}" # Chạy mượt mà vì đã có self.name

    class Meta:
        verbose_name = "Cluster information"
        verbose_name_plural = "Cluster information"

admin.site.register(GenderID)
admin.site.register(BAS_werkzaamheid_resp)
admin.site.register(SPSS_Regio5)
admin.site.register(BAS_bruto_jaarinkomen)
admin.site.register(AFG_sk2015)
admin.site.register(BAS_voltooide_opleiding8_resp)
admin.site.register(SPSS_Lifestage)
admin.site.register(type_touch)
admin.site.register(cluster_info)