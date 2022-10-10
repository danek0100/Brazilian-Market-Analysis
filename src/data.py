from investiny import historical_data
import yfinance as yf
from os import walk
from time import sleep
import pandas as pd
import csv

from stock import Stock


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


def get_Stocks(levelValueAtRisk, ids_path="../resource/brazil_ids.csv"):
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
                stocks[-1].name = row['full_name']

    for stock in stocks:
        stock.profitability = pd.DataFrame(stock.close_price).pct_change()
        stock.E = stock.profitability.mean()[0]
        stock.sigma = stock.profitability.std()[0]
        stock.profitability_sorted = pd.DataFrame(stock.close_price).pct_change()
        for stock_profit in stock.profitability_sorted:
            stock_profit *= -1
        stock.profitability_sorted = stock.profitability_sorted.sort_values(by=[0])

        stock.ValueAtRisk[levelValueAtRisk] = stock.profitability_sorted[0][
                                              int(len(stock.profitability_sorted) *
                                                  (1.0 - float(levelValueAtRisk))):].min()
    return stocks


def get_market_index(key, levelValueAtRisk, from_date="2017-01-01", to_date="2018-01-01"):
    data = yf.download(key, from_date, to_date)
    prices_array = []
    volumes_array = []
    for price in data['Adj Close']:
        prices_array.append(price)
    for volume in data['Volume']:
        volumes_array.append(volume)

    stock = Stock(0, key, prices_array, volumes_array)
    stock.profitability = data['Adj Close'].pct_change()
    stock.profitability_sorted = stock.profitability.copy()
    for stock_proft in stock.profitability_sorted:
        stock_proft *= -1
    stock.ValueAtRisk[levelValueAtRisk] = stock.profitability_sorted.array[int(len(stock.profitability_sorted) *
                                                                               (1.0 - float(levelValueAtRisk))):].min()
    stock.E = stock.profitability.mean()
    stock.sigma = stock.profitability.std()
    return stock