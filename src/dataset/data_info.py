def dataset_info(dataset):
    file_name = dataset.keys()
    print(file_name)
    nb_files = len(file_name)
    print(f"Tổng file: {nb_files}")
    for file in file_name:
        print(f"Tên file: {file}")
        print(f"file info: {dataset[file].info()}")
        print('Dữ liệu mẫu')
        print(dataset[file].head())