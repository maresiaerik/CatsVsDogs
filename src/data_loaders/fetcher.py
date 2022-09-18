import os
import requests
import zipfile


_DATA_URL = "https://unimibox.unimi.it/index.php/s/eNGYGSYmqynNMqF/download"


def fetch_and_save_data():
    zip_folder_name = "CatsDogs.zip"
    data_dir = "data"

    if os.path.isdir(f'./{data_dir}'):
        return

    zip_folder = requests.get(_DATA_URL)

    _save_data_and_unzip(zip_folder_name, zip_folder, data_dir)


def _save_data_and_unzip(zip_folder_name: str, zip_folder: requests.Response, data_dir: str):
    _save_folder(zip_folder_name, zip_folder)
    _unzip_folder(zip_folder_name, data_dir)


def _save_folder(zip_folder_name: str, zip_folder: requests.Response):
    with open(zip_folder_name, "wb") as f:
        f.write(zip_folder.content)


def _unzip_folder(zip_folder_name: str, data_dir: str):
    with zipfile.ZipFile(zip_folder_name, 'r') as zip_ref:
        zip_ref.extractall(f"./{data_dir}")

    os.remove(f"./{zip_folder_name}")
