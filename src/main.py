import os

from data import *
from stock import Stock
from functions import *

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
stocks = get_Stocks(levelValueAtRisk)

# Add mean and var to stocks
Es = []
sigmas = []

for stock in stocks:
    Es.append(stock.E)
    sigmas.append(stock.sigma)

# Add market index
stocks.append(get_market_index('^BVSP', levelValueAtRisk))
Es.append(stocks[-1].E)
sigmas.append(stocks[-1].sigma)


# Add equal backpack
########################################################################################################################
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
print("sds", stocks[-1].E)
print(stocks[-1].sigma)
########################################################################################################################

maxValueAtRiskStock = min(stocks, key=lambda x: x.ValueAtRisk[levelValueAtRisk])
print(maxValueAtRiskStock.key)
print(maxValueAtRiskStock.ValueAtRisk[levelValueAtRisk])

VAR = []
i = 0 
for stock in stocks:
    loss = pd.DataFrame(stock.profitability_sorted)
    VAR.append(loss.quantile(0.95))

minVAR = 1
for i in range(len(VAR) - 1):
    print(VAR[i][0])
    if VAR[i][0] < minVAR:
        minVAR = VAR[i][0]
        minID = i

print("VAR ", minVAR )
print("VAR id ", minID )
print(len(VAR))
i = 0
for stock in stocks:
    if i == minID:
        VARsig = stock.sigma
        VARE = stock.E
        print(stock.name)
    i += 1 

print(stocks[-1].profitability)
 
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
#plt.show()

needed_stocks = ['VALE3', 'ITUB3', 'GOLL4', 'VIVT3', 'CMIG3']

# VALE3 - горная добыча
# ITUB3 - крупнейший негосударственный банк
# GOLL4 - бразильские авиалинии
# BBDC3 - третий по величине банк бразилии
# VIVT3 - оператор сотовой связи
# CMIG3 - электричество
selected_stocks = []
for name_stock in needed_stocks:
    find_stock = next(stock for stock in stocks if stock.key == name_stock)
    selected_stocks.append(find_stock)

alpha = 0.05
print('Критерий инверсии:')
for stock in selected_stocks:
    print('\n' + str(stock.name))
    #print([round(price, 2) for price in stock.profitability[0]])
    #print(stock.volume)
    for target in ['profit', 'volume']:
        result, p_value = inversion_test(stock, alpha, target)
        if result:
            print(f'Г-за случайности отвергается ' + target + ' ' + str(round(p_value, 3)))
        else:
            print(f'Г-за случайности принимается ' + target + ' ' + str(round(p_value, 3)))

get_auto_correlation_plot(selected_stocks)

plot_vs_pdf(selected_stocks)

for stock in selected_stocks:
    print("Для {}:".format(stock.name))
    test_hypothesis(stock)


# Рассматриваем актив, где гипотеза не отвергается
plot_profit(stocks, 'VALE3')
plot_volume(stocks, 'VALE3')
plot_prices(stocks, 'VALE3')

print('\n')
# Корреляция
# ITUB3 - крупнейший негосударственный банк
# BBDC3 - третий по величине банк бразилии
ITUB = next(stock for stock in stocks if stock.key == 'ITUB3')
BBDC = next(stock for stock in stocks if stock.key == 'BBDC3')
corr_data = ITUB.profitability.copy()
corr_data_2 = BBDC.profitability.copy()
corr_data[1] = corr_data_2
print("Корреляция между " + ITUB.name + " и " + BBDC.name + ' = ' + str(round(corr_data.corr()[0][1], 2)))


# Корреляция
# VIVT3 - оператор сотовой связи
# TELB3 - телекоммуникация
VIVT3 = next(stock for stock in stocks if stock.key == 'VIVT3')
TELB4 = next(stock for stock in stocks if stock.key == 'TELB4')
corr_data = VIVT3.profitability.copy()
corr_data_2 = TELB4.profitability.copy()
corr_data[1] = corr_data_2
print("Корреляция между " + VIVT3.name + " и " + TELB4.name + ' = ' + str(round(corr_data.corr()[0][1], 2)))


# Корреляция
# VIVT3 - оператор сотовой связи
# CMIG3 - электроэнергетика
VIVT3 = next(stock for stock in stocks if stock.key == 'VIVT3')
CMIG3 = next(stock for stock in stocks if stock.key == 'CMIG3')
corr_data = VIVT3.profitability.copy()
corr_data_2 = CMIG3.profitability.copy()
corr_data[1] = corr_data_2
print("Корреляция между " + VIVT3.name + " и " + CMIG3.name + ' = ' + str(round(corr_data.corr()[0][1], 2)))


# Корреляция
# VALE3 - горная добыча
# ITUB3 - крупнейший негосударственный банк
VALE3 = next(stock for stock in stocks if stock.key == 'VALE3')
ITUB3 = next(stock for stock in stocks if stock.key == 'ITUB3')
corr_data = VALE3.profitability.copy()
corr_data_2 = ITUB3.profitability.copy()
corr_data[1] = corr_data_2
print("Корреляция между " + VALE3.name + " и " + ITUB3.name + ' = ' + str(round(corr_data.corr()[0][1], 2)))


# Корреляция
# VALE3 - горная добыча
VALE3 = next(stock for stock in stocks if stock.key == 'VALE3')
corr_data = VALE3.profitability.copy()
corr_data_2 = VALE3.volume.copy()
corr_data[1] = corr_data_2
print("Корреляция " + VALE3.name + " между доходностью и объёмом = " + str(round(corr_data.corr()[0][1], 2)))


# Корреляция
# CMIG3 - электричество
CMIG3 = next(stock for stock in stocks if stock.key == 'CMIG3')
corr_data = CMIG3.profitability.copy()
corr_data_2 = CMIG3.volume.copy()
corr_data[1] = corr_data_2
print("Корреляция " + CMIG3.name + " между доходностью и объёмом = " + str(round(corr_data.corr()[0][1], 2)))

get_independent_set(stocks[:-3])
# Новости и интерпретация пиков продаж
plt.show()


# #prepearing fot graphs
df_for_graph = pd.DataFrame(
    {'σ': sigmas[:-2],
     'E': Es[:-2]
    })
#
#
# #for 2 point
sns.set_style("darkgrid")
fig, ax = plt.subplots(1,2)
plt.subplots_adjust(wspace= 0.5)
sns.scatterplot(data = df_for_graph, x='σ', y='E', ax=ax[0]).set_title("Profitability/Risk Map")
ax[0].legend(['Assets'])
df_graph_plus = df_for_graph.drop(np.where(df_for_graph['σ'] > 0.08)[0])
sns.scatterplot(data = df_graph_plus, x='σ', y='E', ax=ax[1]).set_title("Profitability/Risk Map")
ax[1].legend(['Assets'])
fig.show()
#
#
# #for 3 point
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
#
# #for 4 point
sns.set_style("darkgrid")
fig_2, ax = plt.subplots(1,2)
plt.subplots_adjust(wspace= 0.5)

sns.scatterplot(data = df_for_graph, x='σ', y='E', ax=ax[0]).set_title("Profitability/Risk Map")
ax[0].plot(sigmas[-2], Es[-2], color='yellow', marker='o')
ax[0].plot(sigmas[-1], Es[-1], color='green', marker='o')
ax[0].legend(['Assets', 'Balanced portfolio', 'Market index - BOVESPA'])
df_graph_plus = df_for_graph.drop(np.where(df_for_graph['σ'] > 0.08)[0])
sns.scatterplot(data = df_graph_plus, x='σ', y='E', ax=ax[1]).set_title("Profitability/Risk Map")
#plt.plot(maxValueAtRiskStock.sigma, maxValueAtRiskStock.E, color='red', marker='o')

ax[1].plot(sigmas[-1], Es[-1], color='green', marker='o')
ax[1].plot(sigmas[-2], Es[-2], color='yellow', marker='o')
ax[1].legend(['Assets', 'Balanced portfolio', 'Market index - BOVESPA'])
fig_2.show()
#
# #for 5 point
sns.set_style("darkgrid")
fig_3, ax = plt.subplots(1,2)
plt.subplots_adjust(wspace= 0.5)

sns.scatterplot(data = df_for_graph, x='σ', y='E', ax=ax[0]).set_title("Profitability/Risk Map")
ax[0].plot(sigmas[-2], Es[-2], color='yellow', marker='o')
ax[0].plot(sigmas[-1], Es[-1], color='green', marker='o')
ax[0].plot(VARsig, VARE, color='red', marker='o')
ax[0].legend(['Assets', 'Balanced portfolio' , 'Market index - BOVESPA', 'VAR'])

df_graph_plus = df_for_graph.drop(np.where(df_for_graph['σ'] > 0.08)[0])
sns.scatterplot(data = df_graph_plus, x='σ', y='E', ax=ax[1]).set_title("Profitability/Risk Map")
#plt.plot(maxValueAtRiskStock.sigma, maxValueAtRiskStock.E, color='red', marker='o')

ax[1].plot(sigmas[-1], Es[-1], color='green', marker='o')
ax[1].plot(sigmas[-2], Es[-2], color='yellow', marker='o')
ax[1].plot(VARsig, VARE, color='red', marker='o')
ax[1].legend(['Assets', 'Balanced portfolio' , 'Market index - BOVESPA', 'VAR'])
fig_3.show()
input() #to stop close graph:)
#