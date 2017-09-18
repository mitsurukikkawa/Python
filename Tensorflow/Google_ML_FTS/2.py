import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from pandas.plotting import scatter_matrix

# ダウンロードしてきたやつ
INDEIES = ["N225",  # Nikkei 225, Japan
           "HSI",   # Hang Seng, Hong Kong
           "GDAXI", # DAX, German
           "DJI",   # Dow, US
           "GSPC",  # S&P 500, US
           "SSEC",  # Shanghai Composite Index (China)
           "BVSP"]  # BOVESPA, Brazil
def study():
    closing = pd.DataFrame()
    for index in INDEIES:
        # na_valuesは文字列"null"のとき空として扱う CSVみるとnullって書いてあります。
        df = pd.read_csv("./data/" + index + ".csv",na_values=["null"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        closing[index] = df["Close"]
    closing = closing.fillna(method="ffill")
    '''
    print(closing.describe())
    '''
    # グラフ表示
    for index in INDEIES:
        closing[index] = closing[index] / max(closing[index])
        closing[index] = np.log(closing[index] / closing[index].shift())
    closing.plot()
    plt.show()

    #自己相関
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(15)
    for index in INDEIES:
        autocorrelation_plot(closing[index], label=index)
    plt.show()
    '''
    #散布図行列
    scatter_matrix(closing, figsize=(20, 20), diagonal='kde')
    plt.show()
    '''

if __name__ == "__main__":
    study()

