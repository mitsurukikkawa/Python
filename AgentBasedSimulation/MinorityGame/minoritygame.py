import numpy as np
from numpy import random as rnd
from matplotlib import pyplot as plt

N = 1001 # エージェントの数
M = 2 # 戦略表の選択肢の数
T = 1000 # シミュレーションの試行回数
history = np.array([1,1,1])

class Agent(object):
    # グローバル変数として指定するのはイケてない気がする
    global M
    
    # エージェントの内部状態の初期値
    def __init__(self):
        self.strategy_list = self.__make_init_strategy_list()
        self.strategy_points = np.zeros(M)
    
    # 戦略表の初期値
    def __make_init_strategy_list(self):
        def make_one_strategy():
            opt = np.array([-1,1])
            # Pythonの辞書（連想配列）はkeyにリストを持てず、タプルを使う
            return {(x,y,z): np.sign(rnd.random() - 0.5) 
                    for x in opt for y in opt for z in opt}
        return [make_one_strategy() for i in range(M)]
    
    # エージェントの戦略を決定
    def decide_action(self, history):
        hist_key = tuple(history)
        strategy_list = self.__choose_strategy_list()
        strategy = strategy_list[hist_key]
        return strategy
    
    # エージェントが戦略表を選ぶ
    def __choose_strategy_list(self):
        # 戦略表のポイントが最大値のindexを返す
        max_indexes = [i for i,j in enumerate(self.strategy_points) 
                       if j == np.max(self.strategy_points)]
        # 最大値のkeyをランダムに選ぶ
        rnd.shuffle(max_indexes)
        max_index = max_indexes[0]
        return self.strategy_list[max_index]
    
    # 戦略表のポイントをアップデートする
    def update_strategy_points(self, history, headcount):
        hist_key = tuple(history)
        updates = np.array([x[hist_key] for x in self.strategy_list]) * np.sign(headcount)
        self.strategy_points = self.strategy_points - updates

# エージェントをN人用意
agents = [Agent() for i in range(N)]

output = []
for t in range(T):
    # Python3では、np.sumはmap関数の戻り値をちゃんと評価できないらしいのでリスト内包表記を利用
    headcount = np.sum([a.decide_action(history) for a in agents])
    # 合計がちょうど0のとき、headcountの値を+1または-1にする（今回は人数が奇数なのでありえないが）
    if headcount == 0:
        headcount = np.sign(rnd.random() - 0.5)
    for agent in agents:
        agent.update_strategy_points(history, headcount)
    history = np.append(history[1:], np.sign(headcount))
    output.append(0.5 - np.abs(headcount/N)/2)

# csvとかで書き出してもよし、プロットしてもよし
plt.plot(output)
plt.show()
