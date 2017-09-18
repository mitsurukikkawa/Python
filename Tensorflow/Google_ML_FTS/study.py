import numpy as np
import pandas as pd
from model import Model
# ダウンロードしてきたやつ
INDEIES = ["N225",  # Nikkei 225, Japan
           "HSI",   # Hang Seng, Hong Kong
           "GDAXI", # DAX, German
           "DJI",   # Dow, US
           "GSPC"]  # S&P 500, US
'''
           "SSEC",  # Shanghai Composite Index (China)
           "BVSP"]  # BOVESPA, Brazil
'''
def getClosing():
    closing = pd.DataFrame()
    for index in INDEIES:
        # na_valuesは文字列"null"のとき空として扱う CSVみるとnullって書いてあります。
        df = pd.read_csv("./data/" + index + ".csv",na_values=["null"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        closing[index] = df["Close"]
    #空の部分は古いので埋める。
    closing = closing.fillna(method="ffill")
    for index in INDEIES:
        closing[index] = closing[index] / max(closing[index])
        closing[index] = np.log(closing[index] / closing[index].shift())
    closing["positive"] = 0
    #closing["N225"] >= 0の行のpositiveに1をいれる。
    closing.ix[closing["N225"] >= 0, "positive"] = 1
    closing["negative"] = 0
    #closing["N225"] < 0の行のnegativeに1をいれる。
    closing.ix[closing["N225"] < 0, "negative"] = 1
    return closing
def getTraningData():
    closing = getClosing()

    #1~3日前のデータを予測に使う
    days_before = range(1,4)
    answers = pd.DataFrame(columns = ["positive", "negative"])

    columns = []
    for i in days_before :
        columns += [index + "_" + str(i) for index in INDEIES]
    features = pd.DataFrame(columns = columns)
    #なんで7から？
    for i in range(7, len(closing)):
        #予測の部分は当日のデータで
        answers = answers.append({
            "positive" : closing["positive"].ix[i],
            "negative" : closing["negative"].ix[i]}, ignore_index=True)
        data={}
        #ほかの指標は１個前のデータを使用する。
        for index in INDEIES:
            for before in days_before :
                data[index + "_" + str(before)] = closing[index].ix[i - before]
        features = features.append(data, ignore_index=True)
    #予測する元のデータ , 予測するべきデータ
    return features,answers
if __name__ == "__main__":
    features,answers = getTraningData()
    #[]は隠れ層なし、[50.25]は２層
    for layers in [[],[50,25]]:
        model = Model(features,answers,layers)
        model.train(3000)
        print('Accuracy = ',  model.test())
