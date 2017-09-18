import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print(closing.describe())
    # グラフ表示
    closing.plot()
    plt.show()

if __name__ == "__main__":
    study()

