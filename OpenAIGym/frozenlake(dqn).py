import copy, sys
import numpy as np
import matplotlib.pylab as plt
import gym
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers


class Neuralnet(Chain):
    def __init__(self, n_in, n_out):
        super(Neuralnet, self).__init__(
            l1=L.Linear(n_in, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 100),
            q_value=L.Linear(100, n_out, initialW=np.zeros((n_out, 100), dtype=np.float32))
        )

    def Q_func(self, x):
        h = F.leaky_relu(self.l1(x))
        h = F.leaky_relu(self.l2(h))
        h = F.leaky_relu(self.l3(h))
        h = self.q_value(h)
        return h


def main():
    sys.setrecursionlimit(10000)
    env = gym.make("FrozenLake-v0")
    n_obs = env.observation_space.n
    n_act = env.action_space.n
    q = Neuralnet(n_obs, n_act)
    target_q = copy.deepcopy(q)
    optimizer = optimizers.Adam()
    optimizer.setup(q)
    loss = 0
    total_step = 0
    gamma = 0.99
    memory = []
    memory_size = 1000
    batch_size = 100
    epsilon = 1
    epsilon_decrease = 0.005
    epsilon_min = 0
    start_reduce_epsilon = 1000
    train_freq = 10
    update_target_q_freq = 20
    n_epoch = 1000
    n_max_steps = 200
    last_rewards = np.zeros(n_epoch)
    for epoch in range(n_epoch):
        pobs = env.reset()
        pobs = np.identity(n_obs, dtype=np.float32)[pobs, :].reshape((1, n_obs))
        done = False
        for step in range(n_max_steps):
            # select action by epsilon-greedy
            pact = env.action_space.sample()
            if np.random.rand() > epsilon:
                a = q.Q_func(Variable(pobs))
                pact = np.argmax(a.data[0])
            # step
            obs, reward, done, _ = env.step(pact)
            obs = np.identity(n_obs, dtype=np.float32)[obs, :].reshape((1, n_obs))
            # stock experience
            memory.append([pobs, pact, reward, obs, done])
            if len(memory) > memory_size:
                memory.pop(0)
            # train
            if len(memory) >= memory_size:
                if total_step % train_freq == 0:
                    # replay experience
                    np.random.shuffle(memory)
                    memory_idx = range(len(memory))
                    for idx in memory_idx[::batch_size]:
                        batch = memory[idx:idx + batch_size]
                        pobss, pacts, rewards, obss, dones = [], [], [], [], []
                        for b in batch:
                            pobss.append(b[0].tolist())
                            pacts.append(b[1])
                            rewards.append(b[2])
                            obss.append(b[3].tolist())
                            dones.append(b[4])
                        pobss = np.array(pobss, dtype=np.float32)
                        pacts = np.array(pacts, dtype=np.int8)
                        rewards = np.array(rewards, dtype=np.float32)
                        obss = np.array(obss, dtype=np.float32)
                        dones = np.array(dones, dtype=np.bool)
                        Q = q.Q_func(Variable(pobss))
                        tmp = target_q.Q_func(Variable(obss))
                        tmp = list(map(np.max, tmp.data))
                        max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
                        target = np.asanyarray(copy.deepcopy(Q.data), dtype=np.float32)
                        for i in range(batch_size):
                            target[i, pacts[i]] = rewards[i] + gamma * max_Q_dash[i] * (not dones[i])
                        q.zerograds()
                        loss = F.mean_squared_error(Q, Variable(target))
                        loss.backward()
                        optimizer.update()
                if total_step % update_target_q_freq == 0:
                    target_q = copy.deepcopy(q)
            # reduce epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease
            # update last reward
            last_rewards[epoch] = reward
            total_step += 1
            pobs = obs
            if done:
                break
                # print("\t".join(map(str,[epoch, epsilon, loss, reward, total_step])))
    rates = np.average(last_rewards.reshape([n_epoch // 10, 10]), axis=1)
    plt.plot(rates)
    plt.savefig("result.png")


if __name__ == "__main__":
    main()
