import os
import pandas as pd
import math
import matplotlib.pyplot as plt

from src.data import *
from src.stock import Stock

# constants
ids_path = "../resource/brazil_ids.csv"

if not os.path.isfile(ids_path):
    print("Ids for download doesn't found")
    exit(1)

# download data if it will be needed
if not os.path.isdir('../resource/data/'):
    get_data(ids_path)

# create stocks from files with data
stocks = get_Stocks()


# Add mean and var to stocks
Es = []
sigmas = []

for stock in stocks:
    stock.profitability = pd.DataFrame(stock.close_price).pct_change()
    stock.E = stock.profitability.mean()[0]
    stock.sigma = stock.profitability.var()[0]

    Es.append(stock.E)
    sigmas.append(stock.sigma)


# Add market index
stocks.append(get_market_index('^BVSP'))
Es.append(stocks[-1].E)
sigmas.append(stocks[-1].sigma)


# Add equal backpack
sum_Es = 0.0
sum_sigmas = 0.0
for i in range(len(Es)):
    sum_Es += Es[i]
    sum_sigmas += sigmas[i]

stocks.append(Stock(0, "EQAL", [], []))
stocks[-1].sigma = sum_sigmas / len(sigmas)
stocks[-1].E = sum_Es / len(Es)

Es.append(stocks[-1].E)
sigmas.append(stocks[-1].sigma)


# paint map
plt.title('Profitability/Risk Map', fontsize=12, fontname='Times New Roman')
plt.grid(True)
plt.xlabel('sigma', color='gray')
plt.ylabel('E', color='gray')
plt.plot(sigmas[:-2], Es[:-2], 'b*')
# for stock in stocks:
#    plt.text(stock.sigma[0], stock.E[0] + 0.5, stock.key)
plt.plot(sigmas[-2], Es[-2], 'r*')
plt.plot(sigmas[-1], Es[-1], 'g*')
plt.show()


