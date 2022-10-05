from investiny import historical_data
from os import walk
from time import sleep
from src.stock import Stock
import csv


def get_data(ids_path="../resource/brazil_ids.csv", from_date="01/01/2017", to_date="01/01/2018"):
    temp_data = {}
    with open(ids_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                temp_data = historical_data(investing_id=row['id'], from_date=from_date, to_date=to_date)
                sleep(1)
            except Exception:
                continue

            if len(temp_data['close']) > 210:
                with open('../resource/data/' + row['id'] + '.csv', mode='w', encoding='utf-8') as w_file:
                    column_names = ["close", "volume"]
                    file_writer = csv.DictWriter(w_file, delimiter=",",
                                                 lineterminator="\r", fieldnames=column_names)
                    file_writer.writeheader()
                    for i in range(len(temp_data['close'])):
                        file_writer.writerow({"close": str(temp_data['close'][i]), "volume": str(temp_data['volume'][i])})


def get_Stocks(ids_path="../resource/brazil_ids.csv"):
    stocks = []
    ids = []
    files = []
    for (dirpath, dirnames, filenames) in walk("../resource/data"):
        files.extend(filenames)
        break

    for file in files:
        ids.append(int(file[:-4]))
    print(ids)
    print(files)

    with open(ids_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['id']) in ids:
                close = []
                volume = []
                with open('../resource/data/' + row['id'] + '.csv', encoding='utf-8') as data_file:
                    data_reader = csv.DictReader(data_file)
                    for data_row in data_reader:
                        close.append(float(data_row['close']))
                        volume.append(int(data_row['volume']))
                stocks.append(Stock(row['id'], row['symbol'], close, volume))

    return stocks
