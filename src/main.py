from data_loaders.fetcher import fetch_and_save_data
from data_processers.data_processer import process_data_and_retrieve_files_df

from models import AlexNet


def main():
    try:
        fetch_and_save_data()
        files_df = process_data_and_retrieve_files_df()
    except Exception as e:
        print(f"An error occured: {e}")


if __name__ == "__main__":
    main()
