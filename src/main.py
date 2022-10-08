import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

from data import *
from stock import Stock

# constants
ids_path = "../resource/brazil_ids.csv"
levelValueAtRisk = '0.95'


def comp(stock1, stock2):
    if stock1.valueAtRisk[levelValueAtRisk] > stock2.valueAtRisk[levelValueAtRisk]:
        return 1
    else:
        return -1


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
    stock.profitability_sorted = pd.DataFrame(stock.close_price).pct_change().sort_values(by=[0])
    stock.ValueAtRisk[levelValueAtRisk] = stock.profitability_sorted[0][int(len(stock.profitability_sorted) *
                                                                            (1.0 - float(levelValueAtRisk))):].min()
    Es.append(stock.E)
    sigmas.append(stock.sigma)

# Add market index
stocks.append(get_market_index('^BVSP', levelValueAtRisk))
Es.append(stocks[-1].E)
sigmas.append(stocks[-1].sigma)

# Add equal backpack

sum_Es = 0.0
sum_sigmas = 0.0
for i in range(len(Es)):
    sum_Es += Es[i]
    sum_sigmas += sigmas[i]

globalValueAtRisk = (min(sorted(stocks, key=lambda x: x.ValueAtRisk[levelValueAtRisk])
                         [int(len(stocks) * (1.0 - float(levelValueAtRisk))):],
                         key=lambda x: x.ValueAtRisk[levelValueAtRisk])).ValueAtRisk[levelValueAtRisk]
stocks.append(Stock(0, "EQAL", [], []))
stocks[-1].sigma = sum_sigmas / len(sigmas)
stocks[-1].E = sum_Es / len(Es)
stocks[-1].ValueAtRisk[levelValueAtRisk] = globalValueAtRisk
Es.append(stocks[-1].E)
sigmas.append(stocks[-1].sigma)

maxValueAtRiskStock = max(stocks, key=lambda x: x.ValueAtRisk[levelValueAtRisk])
print(maxValueAtRiskStock.key)
print(maxValueAtRiskStock.ValueAtRisk[levelValueAtRisk])

# paint map
plt.title('Profitability/Risk Map', fontsize=12, fontname='Times New Roman')
plt.grid(True)
plt.xlabel('sigma', color='gray')
plt.ylabel('E', color='gray')
plt.plot(sigmas[:-2], Es[:-2], 'b*')
# for stock in stocks:
#    plt.text(stock.sigma[0], stock.E[0] + 0.5, stock.key)
plt.plot(maxValueAtRiskStock.sigma, maxValueAtRiskStock.E, 'y*')
plt.plot(sigmas[-2], Es[-2], 'r*')
plt.plot(sigmas[-1], Es[-1], 'g*')
plt.show()


#prepearing fot graphs
df_for_graph = pd.DataFrame(
    {'σ': sigmas[:-2],
     'E': Es[:-2]
    })


#for 2 point
sns.set_style("darkgrid")
fig, ax = plt.subplots(1,2)
plt.subplots_adjust(wspace= 0.5)
sns.scatterplot(data = df_for_graph, x='σ', y='E', ax=ax[0]).set_title("Profitability/Risk Map")
ax[0].legend(['Assets'])
df_graph_plus = df_for_graph.drop(np.where(df_for_graph['σ'] > 0.08)[0])
sns.scatterplot(data = df_graph_plus, x='σ', y='E', ax=ax[1]).set_title("Profitability/Risk Map")
ax[1].legend(['Assets'])
fig.show()


#for 3 point
sns.set_style("darkgrid")
fig_1, ax = plt.subplots(1,2)
plt.subplots_adjust(wspace= 0.5)

sns.scatterplot(data = df_for_graph, x='σ', y='E', ax=ax[0], legend = "full").set_title("Profitability/Risk Map")
ax[0].plot(sigmas[-1], Es[-1], color='green', marker='o')
ax[0].legend(['Assets', 'Balanced portfolio'])

df_graph_plus = df_for_graph.drop(np.where(df_for_graph['σ'] > 0.08)[0])
sns.scatterplot(data = df_graph_plus, x='σ', y='E', ax=ax[1]).set_title("Profitability/Risk Map")
ax[1].plot(sigmas[-1], Es[-1], color='green', marker='o', label = "Balanced portfolio")
ax[1].legend(['Assets', 'Balanced portfolio'])
fig_1.show()

#for 4 point
sns.set_style("darkgrid")
fig_2, ax = plt.subplots(1,2)
plt.subplots_adjust(wspace= 0.5)

sns.scatterplot(data = df_for_graph, x='σ', y='E', ax=ax[0]).set_title("Profitability/Risk Map")
ax[0].plot(sigmas[-2], Es[-2], color='yellow', marker='o')
ax[0].plot(sigmas[-1], Es[-1], color='green', marker='o')
ax[0].legend(['Assets', 'Market index - BOVESPA','Balanced portfolio'])
df_graph_plus = df_for_graph.drop(np.where(df_for_graph['σ'] > 0.08)[0])
sns.scatterplot(data = df_graph_plus, x='σ', y='E', ax=ax[1]).set_title("Profitability/Risk Map")
#plt.plot(maxValueAtRiskStock.sigma, maxValueAtRiskStock.E, color='red', marker='o')

ax[1].plot(sigmas[-1], Es[-1], color='green', marker='o')
ax[1].plot(sigmas[-2], Es[-2], color='yellow', marker='o')
ax[1].legend(['Assets', 'Market index - BOVESPA','Balanced portfolio'])
fig_2.show()

#for 5 point
sns.set_style("darkgrid")
fig_3, ax = plt.subplots(1,2)
plt.subplots_adjust(wspace= 0.5)

sns.scatterplot(data = df_for_graph, x='σ', y='E', ax=ax[0]).set_title("Profitability/Risk Map")
ax[0].plot(sigmas[-2], Es[-2], color='yellow', marker='o')
ax[0].plot(sigmas[-1], Es[-1], color='green', marker='o')
ax[0].plot(maxValueAtRiskStock.sigma, maxValueAtRiskStock.E, color='red', marker='o')
ax[0].legend(['Assets', 'Market index - BOVESPA','Balanced portfolio' , 'VAR'])

df_graph_plus = df_for_graph.drop(np.where(df_for_graph['σ'] > 0.08)[0])
sns.scatterplot(data = df_graph_plus, x='σ', y='E', ax=ax[1]).set_title("Profitability/Risk Map")
#plt.plot(maxValueAtRiskStock.sigma, maxValueAtRiskStock.E, color='red', marker='o')

ax[1].plot(sigmas[-1], Es[-1], color='green', marker='o')
ax[1].plot(sigmas[-2], Es[-2], color='yellow', marker='o')
ax[1].plot(maxValueAtRiskStock.sigma, maxValueAtRiskStock.E, color='red', marker='o')
ax[1].legend(['Assets', 'Market index - BOVESPA','Balanced portfolio', 'VAR'])
fig_3.show()
input() #to stop close graph:)

