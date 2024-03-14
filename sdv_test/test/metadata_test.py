from sdv.datasets.demo import get_available_demos
import pandas as pd
data = get_available_demos(modality='single_table')
print(data.columns)
print(data.head())

from sdv.datasets.demo import download_demo

data, metadata = download_demo(
    modality='single_table',
    dataset_name='fake_hotel_guests'
)
print(data.head())
print(metadata)

from sdv.datasets.local import load_csvs
# assume that my_folder contains a CSV file named 'guests.csv'
datasets = load_csvs(
    folder_name='my_folder/',
    read_csv_parameters={
        'skipinitialspace': True,
        'encoding': 'utf_8'
    })
data = datasets['guests']
print(data.head())
print('----------------')
data = pd.read_excel('my_folder/example.xlsx')
print(data.head())