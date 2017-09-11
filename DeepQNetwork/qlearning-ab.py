'''
環境の状態は、'A'または'B'からなる長さ8の文字列で表され、
その文字列にはある法則により得点がつけられる。
プレイヤーはその法則についての知識を予め持たないが、
文字列中の任意の1文字を選んで'A'または'B'に置き換えることができ、
その結果、その操作による得点の変化量を報酬として受け取る。
https://zaburo-ch.github.io/post/q-learning/
'''
import numpy as np
import matplotlib.pyplot as plt

STR_LEN = 8
ALPHA = 0.1
GAMMA = 0.99
EPS = 0.1

def int_to_state(n):
    ret = ""
    for i in range(STR_LEN):
        if (n >> i) & 1:
            ret += "B"
        else:
            ret += "A"
    return ret


def score(s):
    point_dict = {"A": 1, "BB": 1, "AB": 2, "ABB": 3, "BBA": 3, "BBBB": 4}
    ret = 0
    for i in range(STR_LEN):
        for j in range(i, STR_LEN):
            if s[i:j+1] in point_dict:
                ret += point_dict[s[i:j+1]]
    return ret


def do_action(s, a):
    current_score = score(s)
    if a == 0:
        next_s = s
    elif 1 <= a <= STR_LEN:
        next_s = s[:a-1] + "A" + s[a:]
    else:
        next_s = s[:a-STR_LEN-1] + "B" + s[a-STR_LEN:]
    reward = score(next_s) - current_score
    return next_s, reward


def QLearning(n_rounds, t_max):
    # init
    s0 = "A" * STR_LEN
    A = range(STR_LEN*2+1)
    Q = {}
    for i in range(2**STR_LEN):
        Q[int_to_state(i)] = np.random.random(len(A))
    score_history = []

    # learn
    for i in range(n_rounds):
        s = s0
        for t in range(t_max):
            # select action
            if np.random.random() < EPS:
                a = Q[s].argmax()
            else:
                a = np.random.choice(A)

            # do action
            next_s, reward = do_action(s, a)

            # update Q, and change s
            Q[s][a] += ALPHA * (reward + GAMMA*Q[next_s].max() - Q[s][a])
            s = next_s
        print("round {0:4d}: {1} = {2:2d}".format(i, s, score(s)),)

        # test
        s = s0
        for t in range(t_max):
            a = Q[s].argmax()
            s, _ = do_action(s, a)
        score_history.append(score(s))
        print("test result: {0} = {1}".format(s, score_history[i]))

    # visualize
    plt.plot(score_history)
    plt.show()

if __name__ == '__main__':
    np.random.seed(123)
    QLearning(1000, 20)
