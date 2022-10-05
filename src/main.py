import os
from src.data import *
from src.stock import Stock

ids_path = "../resource/brazil_ids.csv"


if not os.path.isfile(ids_path):
    print("Ids for download doesn't found")
    exit(1)


if not os.path.isdir('../resource/data/'):
    get_data(ids_path)

stocks = get_Stocks()
print(stocks)
