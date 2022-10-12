from turtle import shape
from scipy import stats
from pandas.plotting import autocorrelation_plot
from scipy.stats import anderson, normaltest, ttest_1samp
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import networkx as nx
import pandas as pd
from networkx.algorithms.approximation import clique


# Печать карты (риск, доходность).
def plot_map(Es, risks, target=1, VaR_stock=None):
    # Подготовка датафрейма для построения карты
    df_for_graph = pd.DataFrame(
        {'σ': risks[:-2],
         'E': Es[:-2]
         })

    # Зададим наборы строк для таргетов
    point_data = {1: ['Assets'],
                  2: ['Assets', 'Balanced portfolio'],
                  3: ['Assets', 'Balanced portfolio', 'Market index - BOVESPA'],
                  4: ['Assets', 'Balanced portfolio', 'Market index - BOVESPA', 'VAR']}

    # Распечаетем набор точек на графике в зависимости от выбранного уровня.
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.5)
    sns.scatterplot(data=df_for_graph, x='σ', y='E', ax=ax[0]).set_title("Profitability/Risk Map")
    ax[0].legend(point_data[target])
    if target > 1:
        ax[0].plot(risks[-1], Es[-1], color='green', marker='o')
        ax[1].plot(risks[-1], Es[-1], color='green', marker='o', label="Balanced portfolio")
    if target > 2:
        ax[0].plot(risks[-2], Es[-2], color='yellow', marker='o')
        ax[1].plot(risks[-2], Es[-2], color='yellow', marker='o')
    if target > 3:
        ax[0].plot(VaR_stock.risk, VaR_stock.E, color='red', marker='o')
        ax[1].plot(VaR_stock.risk, VaR_stock.E, color='red', marker='o')

    df_graph_plus = df_for_graph.drop(np.where(df_for_graph['σ'] > 0.08)[0])
    sns.scatterplot(data=df_graph_plus, x='σ', y='E', ax=ax[1]).set_title("Profitability/Risk Map")
    ax[1].legend(point_data[target])
    fig.show()


# Инициирующая функция для сортировки слиянием.
def mergeSort(arr, n):
    temp_arr = [0] * n
    return _mergeSort(arr, temp_arr, 0, n - 1)


# Обычая сортировка слиянием c подсчтётом инверсий
def _mergeSort(arr, temp_arr, left, right):
    inv_count = 0
    if left < right:
        mid = (left + right) // 2
        inv_count += _mergeSort(arr, temp_arr, left, mid)
        inv_count += _mergeSort(arr, temp_arr, mid + 1, right)
        inv_count += merge(arr, temp_arr, left, mid, right)
    return inv_count


# Обычное слияние с подсчётом инверсий.
def merge(arr, temp_arr, left, mid, right):
    i = left
    j = mid + 1
    k = left
    inv_count = 0
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            k += 1
            i += 1
        else:
            # Инверсия #
            temp_arr[k] = arr[j]
            inv_count += (mid - i + 1)
            k += 1
            j += 1
    while i <= mid:
        temp_arr[k] = arr[i]
        k += 1
        i += 1
    while j <= right:
        temp_arr[k] = arr[j]
        k += 1
        j += 1
    for loop_var in range(left, right + 1):
        arr[loop_var] = temp_arr[loop_var]

    return inv_count


# Тест на случайность основанный на инверсиях.
def inversion_test(stock, alpha, target):
    n = 0
    inversion_number = 0
    if target == 'profit':
        n = len(stock.profitability)
        inversion_number = mergeSort(stock.profitability.copy()[0], n)
    elif target == 'volume':
        n = len(stock.volume)
        inversion_number = mergeSort(stock.volume.copy(), n)

    inversion_number_expectation = (n * (n - 1)) / 4
    inversion_number_variance = (n * (n - 1) * (2 * n + 5)) / 72
    normalized_inversion_statistic = (inversion_number - inversion_number_expectation) / (
                inversion_number_variance ** (1 / 2))
    p_value = stats.norm.sf(abs(normalized_inversion_statistic)) * 2
    return abs(normalized_inversion_statistic) >= stats.norm.ppf(1 - alpha / 2), p_value


# Тест на случайность основанный на автокорреляции.
def auto_correlation_test(stock, alpha, target):
    n = 0
    stock_data = []
    if target == 'profit':
        n = len(stock.profitability)
        stock_data = stock.profitability.copy()[0]
        stock_data[0] = stock_data[1]
    elif target == 'volume':
        n = len(stock.volume)
        stock_data = stock.volume.copy()

    sum_1, sum_2, sum_3 = 0, 0, 0
    for i in range(n - 1):
        sum_1 += stock_data[i] * stock_data[i + 1]

    for i in range(n):
        sum_2 += stock_data[i]
        sum_3 += stock_data[i] * stock_data[i]

    r_1_n = (n * sum_1 - sum_2 + n * stock_data[0] * stock_data[n - 1]) / (n * sum_3 - sum_2)
    expect_r_1_n = - 1 / (n - 1)
    variance_r_1_n = (n * (n - 3)) / ((n + 1) * (n - 1) ** 2)
    r_1_n_normalized = (r_1_n - expect_r_1_n) / math.sqrt(variance_r_1_n)
    p_value = stats.norm.sf(abs(r_1_n_normalized)) * 2

    if abs(r_1_n_normalized) >= stats.norm.ppf(1 - alpha / 2):
        print('Г-за случайности отвергается для ' + target + ' - p_value ' +
              str(round(p_value, 3)) + ' по критерию автокорреляции')
    else:
        print('Г-за случайности принимается для ' + target + ' - p_value ' +
              str(round(p_value, 3)) + ' по критерию автокорреляции')


# Рисование графиков автокорреляции и добавление графиков автокорреляции.
def get_auto_correlation_plot(selected_stocks):
    print('\n')
    alpha = 0.05
    for stock in selected_stocks:
        print('Компания: ' + str(stock.name) + '. E = ' + str(round(stock.E, 3)) + '; Risk = ' +
              str(round(stock.risk, 3)) + ';')

        for target in ['profit', 'volume']:
            auto_correlation_test(stock, alpha, target)
            plt.figure(figsize=(8, 6))
            if target == 'profit':
                for_auto_plot = stock.profitability.copy()
                for_auto_plot[0][0] = for_auto_plot[0][1]
                autocorrelation_plot(for_auto_plot[0])
            elif target == 'volume':
                autocorrelation_plot(stock.volume)
            plt.title(f"График автокорреляции для {stock.name} для {target}", size=16)
        print()


def plot_volume(stocks, key):
    plt.figure(figsize=(8, 6))
    stock = next(stock for stock in stocks if stock.key == key)
    plt.plot([i for i in range(len(stock.volume))], stock.volume)
    plt.title(stock.name + ' Volume', size=15)


def plot_profit(stocks, key):
    plt.figure(figsize=(8, 6))
    stock = next(stock for stock in stocks if stock.key == key)
    cp = stock.profitability.copy()
    cp[0][0] = cp[0][1]
    plt.plot([i for i in range(len(stock.profitability))], cp[0])
    plt.title(stock.name + ' Profit', size=15)


def plot_prices(stocks, key):
    plt.figure(figsize=(8, 6))
    stock = next(stock for stock in stocks if stock.key == key)
    plt.plot([i for i in range(len(stock.close_price))], stock.close_price)
    plt.title(stock.name + ' Close prices', size=15)


def plot_vs_pdf(stocks):
    
    for stock in stocks:
        plt.figure(figsize=(8, 6))
        plt.grid()
        cp = stock.profitability.copy()
        cp[0][0] = cp[0][1]
        sns.distplot(cp[0], bins=10)
        #sns.displot(data=cp[0], kde=True)
        plt.title("Плотность доходностей для {}".format(stock.name))


# Тесты для проверки гипотезы о нормальном распределении и распределении Стьюдента.
def test_hypothesis(selected_stock, alpha=0.05):
    cp = selected_stock.profitability.copy()
    cp[0][0] = cp[0][1]

    # Тест Андерсона
    result_a = anderson(cp[0])
    statistic = result_a[0]
    answer = 'отклоняется' if statistic > result_a[1][2] else 'не отвергается'
    print("\t Гипотеза {} {}, статистика={:3f}".format(answer, "теста Anderson'а", statistic))

    # Тест D'Agostino
    result = normaltest(cp[0])
    p_value = result[1]
    answer = 'не отвергается' if p_value > alpha else 'отклоняется'
    print("\t Гипотеза {} {}, p-value={:3f}".format(answer, "тест D'Agostino", p_value))

    # t - тест
    result_t = stats.ttest_1samp(cp[0], selected_stock.E)
    p_value_t = result_t[1]
    answer = 'не отвергается' if p_value_t > alpha else 'отклоняется'
    print("\t Гипотеза {} {}, p-value={:3f}".format(answer, "t тест", p_value_t))


def graph_by_matrix(corr_m, thr):
    G = nx.from_numpy_matrix(np.asmatrix(corr_m))
    edge_weights = nx.get_edge_attributes(G, 'weight')
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_edges_from((e for e, w in edge_weights.items() if w < thr))
    return G


def graph_by_matrix_independent(corr_m, thr):
    G = nx.from_numpy_matrix(np.asmatrix(corr_m))
    edge_weights = nx.get_edge_attributes(G, 'weight')
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_edges_from((e for e, w in edge_weights.items() if w >= thr))
    return G


# Получение независимого множества и графа рынка.
def get_independent_set(stocks):
    # Формируем датафрейм
    d = {}
    stock_names = []
    min_data_len = len(min(stocks[:-3], key=lambda x: len(x.profitability[0])).profitability[0])
    for stock in stocks[:-3]:
        try:
            d[stock.key] = stock.profitability[0].copy()[:min_data_len-1]
            stock_names.append(stock.key)
        except Exception:
            continue
    returns = pd.DataFrame(data=d)
    corr = returns.corr()

    # Строим грфик распределения корреляций между доходносятми активов
    cor_rasp = []
    kf = -1.0
    while kf < 1.0:
        ct = 0
        for stock_i in stock_names:
            for stock_j in stock_names:
                if (kf + 0.1) > corr[stock_i][stock_j] > (kf - 0.1):
                    ct = ct + 1
        cor_rasp.append(ct)
        kf = kf + 0.1

    print("Распределение рёбер:\n" + str(cor_rasp))

    max_cor_rasp = max(cor_rasp)
    cor_rasp.pop()
    cor_rasp.append(0)
    rng = cor_rasp
    rnd = []
    for i in range(-10, 11):
        rnd.append(i / 10)

    plt.figure(figsize=(22, 12))
    plt.axis([-1.1, 1.1, 0, max_cor_rasp + 0.2 * max_cor_rasp])
    plt.title('Number of links', fontsize=20, fontname='Times New Roman')
    plt.xlabel('The correlation coefficient', color='gray')
    plt.ylabel('Number of links', color='gray')
    plt.plot(rnd, rng, 'b-o', alpha=0.8, label="Number of links", lw=5, mec='b', mew=2, ms=5)
    plt.legend()
    plt.grid(True)
    plt.legend(loc='upper left')

    # Строим граф на уровне корреляции 0.8
    for por in range(8, 9):
        pr = por / 10.0
        plt.figure(figsize=(50, 25))
        G = graph_by_matrix(corr, pr)
        GI = graph_by_matrix_independent(corr, pr)

        label_dict = {}
        i = 0
        for name in stock_names[:-3]:
            label_dict[i] = name
            i += 1

        nx.draw_random(G, node_color='blue', node_size=550, with_labels=True, alpha=0.55, width=0.9, font_size=7.5,
                       font_color='black', font_weight='normal', font_family='Times New Roman', labels=label_dict)

        clique_set = clique.max_clique(G)
        print("Максимальная клика:\n" + str(clique_set))
        for node in clique_set:
            print(label_dict[node])
        independent_set = clique.max_clique(GI)
        print("Максимальное независимое множество:\n" + str(independent_set))


def print_corr(stocks, stock_key_1, stock_key_2):
    stock_1 = next(stock for stock in stocks if stock.key == stock_key_1)
    stock_2 = next(stock for stock in stocks if stock.key == stock_key_2)
    corr_data = stock_1.profitability.copy()
    #corr_data_print = stock_1.profitability.copy()
    corr_data_2 = stock_2.profitability.copy()
    #corr_data[1] = corr_data_2

    data_1 = np.squeeze(np.asarray(corr_data[1:]))
    data_2 = np.squeeze(np.asarray(corr_data_2[1:]))
    corr = stats.pearsonr(data_1, data_2)

    print('Корреляция между ' + stock_1.name + '(' + stock_1.key + ')' + ' и ' + stock_2.name + '(' +
          stock_2.key + ')' + ' = ' + str(round(corr[0], 2)) + ' p-value ' + str(corr[1])) 
    scatter_draw(corr_data[1:], corr_data_2[1:], [stock_1.name, stock_2.name, "Кореляция между двумя активами"])
    



def print_corr_profit_volume(stocks, stock_key):
    stock = next(stock for stock in stocks if stock.key == stock_key)
    corr_data = stock.profitability.copy()
    #corr_data_print = stock.profitability.copy()
    corr_data_2 = stock.volume.copy()
    #corr_data[1] = corr_data_2

    data_1 = np.squeeze(np.asarray(corr_data[1:]))
    data_2 = np.squeeze(np.asarray(corr_data_2[1:]))
    corr = stats.pearsonr(data_1, data_2)

    print('Корреляция между ' + stock.name + '(' + stock.key + ')' + ' = ' + str(round(corr[0], 2)) + ' p-value ' + str(corr[1]))
    scatter_draw(corr_data, corr_data_2, ['Profitability','Volume', stock_key])
    

def scatter_draw(stock1, stock2, keys):
    plt.grid(True)
    plt.scatter(stock1, stock2, edgecolors = 'white')
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])
    plt.title(keys[2])
    plt.show()