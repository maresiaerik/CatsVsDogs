import pandas as pd
from sklearn.model_selection import train_test_split

from models import Model
from data_loaders.fetcher import fetch_and_save_data
from data_processers.data_processer import retrieve_dict_of_processed_files


def prepare_data() -> pd.DataFrame:
    fetch_and_save_data()

    return retrieve_dict_of_processed_files()
