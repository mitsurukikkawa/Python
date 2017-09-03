# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def main():
    # アイリスデータを取得
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # ニューラルネットで学習
    clf = MLPClassifier(solver="sgd", random_state=0, max_iter=10000)
    # 予測
    clf.fit(X_train, y_train)
    # 結果表示
    print("識別率：", clf.score(X_test, y_test))


if __name__ == "__main__":
    main()
