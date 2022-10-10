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
from scipy.stats import multivariate_normal
from scipy.linalg import eigh, cholesky


def mergeSort(arr, n):
    temp_arr = [0] * n
    return _mergeSort(arr, temp_arr, 0, n - 1)


def _mergeSort(arr, temp_arr, left, right):
    inv_count = 0
    if left < right:
        mid = (left + right) // 2
        inv_count += _mergeSort(arr, temp_arr, left, mid)
        inv_count += _mergeSort(arr, temp_arr, mid + 1, right)
        inv_count += merge(arr, temp_arr, left, mid, right)
    return inv_count


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
            # Инверсия
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
    normalized_inversion_statistic = (inversion_number - inversion_number_expectation) / (inversion_number_variance ** (1 / 2))
    p_value = stats.norm.sf(abs(normalized_inversion_statistic)) * 2
    return abs(normalized_inversion_statistic) >= stats.norm.ppf(1 - alpha / 2), p_value


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
        print(
            'Г-за случайности ' + stock.name + ' отвергается - p_value ' +
            str(round(p_value, 3)) + ' по критерию автокорреляции ' + target)
    else:
        print(
            'Г-за случайности ' + stock.name + ' принимается - p_value ' +
            str(round(p_value, 3)) + ' по критерию автокорреляции ' + target)

    #  Нормализующим преобразованием статистики этого критерия является статистика Морана

    # Статистика Морана
    r_1_n_moran = math.sqrt((n - 1)) * (n * r_1_n + 1) / (n - 2)
    p_value = stats.norm.sf(abs(r_1_n_moran)) * 2
    if abs(r_1_n_moran) >= stats.norm.ppf(1 - alpha / 2):
        print(
            'Г-за случайности ' + stock.name + ' отвергается - p_value ' +
            str(round(p_value, 3)) + ' по критерию автокорреляции (Морана) ' + target)
    else:
        print(
            'Г-за случайности ' + stock.name + ' принимается - p_value ' +
            str(round(p_value, 3)) + ' по критерию автокорреляции (Морана) ' + target)


def get_auto_correlation_plot(selected_stocks):
    print('\n')
    alpha = 0.05
    for stock in selected_stocks:
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
    #stock.volume.plot(y='Volume', grid=True, figsize=(16, 6))
    plt.title(stock.name + ' Volume', size=15)


def plot_profit(stocks, key):
    plt.figure(figsize=(8, 6))
    stock = next(stock for stock in stocks if stock.key == key)
    cp = stock.profitability.copy()
    cp[0][0] = cp[0][1]
    plt.plot([i for i in range(len(stock.profitability))], cp[0])
    #stock.volume.plot(y='Volume', grid=True, figsize=(16, 6))
    plt.title(stock.name + ' Profit', size=15)


def plot_prices(stocks, key):
    plt.figure(figsize=(8, 6))
    stock = next(stock for stock in stocks if stock.key == key)
    plt.plot([i for i in range(len(stock.close_price))], stock.close_price)
    #stock.volume.plot(y='Volume', grid=True, figsize=(16, 6))
    plt.title(stock.name + ' Close prices', size=15)


def plot_vs_pdf(stocks):
    for stock in stocks:
        plt.figure(figsize=(8, 6))
        plt.grid()
        cp = stock.profitability.copy()
        cp[0][0] = cp[0][1]
        sns.distplot(cp[0], bins=10)
        plt.title("для {}".format(stock.name))


def test_hypothesis(selected_stock, alpha=0.05):
    cp = selected_stock.profitability.copy()
    cp[0][0] = cp[0][1]

    # Anderson test
    result_a = anderson(cp[0])
    statistic = result_a[0]
    answer = 'отклоняется' if statistic > result_a[1][2] else 'не отвергается'
    print("\t Гипотеза {} {}, статистика={:3f}".format(answer, "Anderson-test", statistic))

    # Normal test
    result = normaltest(cp[0])
    p_value = result[1]
    answer = 'не отвергается' if p_value > alpha else 'отклоняется'
    print("\t Гипотеза {} {}, p-value={:3f}".format(answer, "D'Agostino-test",  p_value))

    # T-test
    result_t = stats.ttest_1samp(cp[0], selected_stock.E)
    p_value_t = result_t[1]
    answer = 'не отвергается' if p_value_t > alpha else 'отклоняется'
    print("\t Гипотеза {} {}, p-value={:3f}".format(answer, "T-Test", p_value_t))


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


def get_independent_set(stocks):
    results_for_par = []
    results_for_par_independent_set = []

    d = {}
    stock_names = []
    for stock in stocks:
        try:
            d[stock.key] = stock.profitability[0].copy()
            stock_names.append(stock.key)
        except Exception:
            continue
    returns = pd.DataFrame(data=d)

    corr = returns.corr()
    cov = returns.cov()

    #График распределения кф
    num_stocks = len(stocks)
    cor_rasp = []
    kf = -1.0
    ct = 0

    while kf < 1.0:
        ct = 0
        for stock_i in stock_names:
            for stock_j in stock_names:
                if (kf + 0.1) > corr[stock_i][stock_j] > (kf - 0.1):
                    ct = ct + 1
        cor_rasp.append(ct)
        kf = kf + 0.1

    print(cor_rasp)

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
    plt.savefig('my_chart1.png')

    for por in range(8, 9):
        results = []
        results_independent = []
        pr = por/10.0
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
        print(clique_set)
        independent_set = clique.max_clique(GI)
        print(independent_set)
