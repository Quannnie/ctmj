import requests
import pandas as pd

def get_data(api_url):
    """
    param: api_url
    return: {"file_name":pandas dataframe}
    """
    dataframes = {}

    #request
    response = requests.get(api_url)
    if response.status_code == 200:
        article_data = response.json()
        files = article_data.get('files', [])

        #browse file
        for i, file_info in enumerate(files):
            file_name = file_info['name']
            download_url = file_info['download_url']
            try:

                #convert to dataframe
                if file_name.endswith('.csv'):
                    df = pd.read_csv(download_url)
                    dataframes[file_name] = df
                elif file_name.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(download_url)
                    dataframes[file_name] = df
                else:

                    #can not read
                    print(f"Định dạng {file_name} cần thư viện chuyên dụng để đọc.")

            #error handling
            except Exception as e:
                print(f"Lỗi khi đọc file {file_name}: {e}")
    else:
        print(f"Không thể kết nối với API Figshare. {response.status_code}")

    return dataframes