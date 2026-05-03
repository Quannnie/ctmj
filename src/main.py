#import package
import pandas as pd
from src.dataset import loader, data_info
from src.eda import data_profiling
from src.dictionaries import dictionary
from src.pre_processing import fillna_value

#Get data
print("Getting data from API")
article_id = "23690811"
api_url = f"https://api.figshare.com/v2/articles/{article_id}"
dataset = loader.get_data(api_url)
data_info.dataset_info(dataset)

TravelDataUsers = dataset["B_TravelDataUsers.csv"]
TravelDataJourneys = dataset["A_TravelDataJourneys.csv"]

print('All dataset are ready for use')

trigger_profiling = input('Do you want to get profiling data?(y/n)')

if trigger_profiling =='y':
    # User Dashboard information
    output_dir = "../resources/profiling"
    user_output_filename = "TravelDataUsers_data_profiling.html"
    user_dashboard_title = "Travel Users Dataset Profiling"

    #Run
    data_profiling(TravelDataUsers, output_dir,user_output_filename, user_dashboard_title)
    print('Done profiling for TravelDataUsers')

    #Journey Dashboard information
    output_dir = "../resources/profiling"
    journey_output_filename = "TravelDataJourneys_data_profiling.html"
    journey_dashboard_title = "Travel Journeys Dataset Profiling"

    #Run
    data_profiling(TravelDataJourneys, output_dir,journey_output_filename, journey_dashboard_title)
    print('Done profiling for TravelDataJourneys')
else:
    print('Nothing to do!')

#Preparing data for Clustering
dictionary = dictionary()
touchpoint_mapping = dictionary.type_touch


#Preparing data for Clustering
user_data = TravelDataUsers[~TravelDataUsers['SPSS_Regio5'].isnull()]
selected_column = [
    'UserID',
    'GenderID', #gioi tinh
    'Age', #tuoi
    'SPSS_Regio5', #vung
    'BAS_huishoudgrootte', #quy mo ho gia dinh (so nguoi trong nha)
    'BAS_werkzaamheid_resp', #nghe nghiep
    'BAS_bruto_jaarinkomen', #tong thu nhap truoc thue
    'afg_kinderen_huishouden', #so luong tre em trong gia dinh
    'AFG_sk2015', #tang lop xa hoi
    'BAS_voltooide_opleiding8_resp', #trinh do hoc van
    'SPSS_Lifestage' #giai doan cuoc doi
]

user_data = user_data[selected_column]
user_data = fillna_value(user_data, ["BAS_voltooide_opleiding8_resp"], 97)

#user_data info
print(user_data.info())
print(user_data.head())

#Preparing data for Prediction
#Dictionary mapping
journey_data = TravelDataJourneys[TravelDataJourneys['Duration'] > 0].copy()
#Duration > 0
journey_data['TIMESPSS'] = pd.to_datetime(journey_data['TIMESPSS'])
journey_data = journey_data.sort_values(by=['UserID', 'TIMESPSS']).reset_index(drop=True)

#Đánh dấu các click liên tiếp cùng loại của cùng 1 User
change_mask = (journey_data['type_touch'] != journey_data['type_touch'].shift()) | (journey_data['UserID'] != journey_data['UserID'].shift())
journey_data['group_id'] = change_mask.cumsum()

#Aggregation
journey_data_compressed = journey_data.groupby(['UserID', 'group_id', 'type_touch']).agg({
    'TIMESPSS': 'first',
    'Duration': 'sum',
    'PurchaseID': 'first',
    'DEVICE_TYPE': 'first',
    'purchase_own': 'max',
    'purchase_any': 'max',
    'MobilePanel': 'first',
    'FixedPanel': 'first'
}).reset_index()

# 5. Mapping tên điểm chạm
journey_data_compressed['touchpoint_name'] = journey_data_compressed['type_touch'].map(touchpoint_mapping)

user_id_list = user_data['UserID'].unique()
journey_data_compressed = journey_data_compressed[journey_data_compressed['UserID'].isin(user_id_list)]

# # 6. Dọn dẹp và kiểm tra
journey_data_compressed = journey_data_compressed.drop(columns='group_id')

print(f"Số dòng sau khi nén: {len(journey_data_compressed)}")

#journey_data info
print(journey_data_compressed.info())
print(journey_data_compressed.head())

#Export Journey data
trigger_export_data = input('Do you want to get journey data?(y/n)')

if trigger_export_data =='y':
    journey_data_compressed[['UserID', 'touchpoint_name', 'TIMESPSS']].to_csv("D:/cjm.csv", index=False)
    print('Journey data has been exported to D:/cjm.csv')
else:
    print('Nothing to do!')