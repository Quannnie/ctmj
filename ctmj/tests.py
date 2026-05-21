from django.test import TestCase
from .models import (
    GenderID, BAS_werkzaamheid_resp, SPSS_Regio5, BAS_bruto_jaarinkomen,
    AFG_sk2015, BAS_voltooide_opleiding8_resp, SPSS_Lifestage, type_touch
)

class GenderIDTestCase(TestCase):
    def setUp(self):
        GenderID.objects.create(gender_code=1, gender_name="Nam")
    
    def test_read(self):
        obj = GenderID.objects.get(gender_code=1)
        self.assertEqual(obj.gender_name, "Nam")
        
    def test_update(self):
        obj = GenderID.objects.get(gender_code=1)
        obj.gender_name = "Nữ"
        obj.save()
        updated = GenderID.objects.get(gender_code=1)
        self.assertEqual(updated.gender_name, "Nữ")
        
    def test_delete(self):
        obj = GenderID.objects.get(gender_code=1)
        obj.delete()
        self.assertEqual(GenderID.objects.count(), 0)

class BAS_werkzaamheid_respTestCase(TestCase):
    def setUp(self):
        BAS_werkzaamheid_resp.objects.create(code=1, name="Student")
        
    def test_crud(self):
        # Read
        obj = BAS_werkzaamheid_resp.objects.get(code=1)
        self.assertEqual(obj.name, "Student")
        # Update
        obj.name = "Worker"
        obj.save()
        self.assertEqual(BAS_werkzaamheid_resp.objects.get(code=1).name, "Worker")
        # Delete
        obj.delete()
        self.assertEqual(BAS_werkzaamheid_resp.objects.count(), 0)

class SPSS_Regio5TestCase(TestCase):
    def setUp(self):
        SPSS_Regio5.objects.create(code=1, name="North")
        
    def test_crud(self):
        # Read
        obj = SPSS_Regio5.objects.get(code=1)
        self.assertEqual(obj.name, "North")
        # Update
        obj.name = "South"
        obj.save()
        self.assertEqual(SPSS_Regio5.objects.get(code=1).name, "South")
        # Delete
        obj.delete()
        self.assertEqual(SPSS_Regio5.objects.count(), 0)

class BAS_bruto_jaarinkomenTestCase(TestCase):
    def setUp(self):
        BAS_bruto_jaarinkomen.objects.create(code=1, name="Low")
        
    def test_crud(self):
        obj = BAS_bruto_jaarinkomen.objects.get(code=1)
        self.assertEqual(obj.name, "Low")
        obj.name = "High"
        obj.save()
        self.assertEqual(BAS_bruto_jaarinkomen.objects.get(code=1).name, "High")
        obj.delete()
        self.assertEqual(BAS_bruto_jaarinkomen.objects.count(), 0)

class AFG_sk2015TestCase(TestCase):
    def setUp(self):
        AFG_sk2015.objects.create(code=1, name="A")
        
    def test_crud(self):
        obj = AFG_sk2015.objects.get(code=1)
        self.assertEqual(obj.name, "A")
        obj.name = "B"
        obj.save()
        self.assertEqual(AFG_sk2015.objects.get(code=1).name, "B")
        obj.delete()
        self.assertEqual(AFG_sk2015.objects.count(), 0)

class BAS_voltooide_opleiding8_respTestCase(TestCase):
    def setUp(self):
        BAS_voltooide_opleiding8_resp.objects.create(code=1, name="Primary")
        
    def test_crud(self):
        obj = BAS_voltooide_opleiding8_resp.objects.get(code=1)
        self.assertEqual(obj.name, "Primary")
        obj.name = "Secondary"
        obj.save()
        self.assertEqual(BAS_voltooide_opleiding8_resp.objects.get(code=1).name, "Secondary")
        obj.delete()
        self.assertEqual(BAS_voltooide_opleiding8_resp.objects.count(), 0)

class SPSS_LifestageTestCase(TestCase):
    def setUp(self):
        SPSS_Lifestage.objects.create(code=1, name="Young")
        
    def test_crud(self):
        obj = SPSS_Lifestage.objects.get(code=1)
        self.assertEqual(obj.name, "Young")
        obj.name = "Old"
        obj.save()
        self.assertEqual(SPSS_Lifestage.objects.get(code=1).name, "Old")
        obj.delete()
        self.assertEqual(SPSS_Lifestage.objects.count(), 0)

class type_touchTestCase(TestCase):
    def setUp(self):
        type_touch.objects.create(code=1, name="Website")
        
    def test_crud(self):
        obj = type_touch.objects.get(code=1)
        self.assertEqual(obj.name, "Website")
        obj.name = "App"
        obj.save()
        self.assertEqual(type_touch.objects.get(code=1).name, "App")
        obj.delete()
        self.assertEqual(type_touch.objects.count(), 0)
