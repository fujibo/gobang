import chainer
from chainer import Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np

# http://qiita.com/ashitani/items/1dc0a54da218ec224ad8

def func(x):
    'function for approximation'
    return np.sqrt(x)

def get_batch(n):
    'get n size batch'
    x = np.random.rand(n)
    return (x, func(x))

class MyChain(Chain):
    """docstring for MyChain"""
    def __init__(self):
        # 1 => 16 => 32 => 1
        super(MyChain, self).__init__(
            l1=L.Linear(1, 32),
            l2=L.Linear(32, 64),
            l3=L.Linear(64, 1),
        )
    def __call__(self, x, y):
        return F.mean_squared_error(self.predict(x), y)

    def predict(self, x):
        'return predict value only used in this NN'
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return h3

    def get(self, x):
        'return predict value by float'
        x = np.array([x]).astype(np.float32).reshape(1, 1)
        return self.predict(Variable(x)).data[0][0]

        self.predict(Variable(np.array([x]).astype(np.float32).reshape(1,1))).data[0][0]


if __name__ == '__main__':
    model = MyChain()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    losses = []

    plt.hold(False)

    for i in range(500):
        x, y = get_batch(100)
        x_ = Variable(x.astype(np.float32).reshape(100, 1))
        y_ = Variable(y.astype(np.float32).reshape(100, 1))

        # 積算するので
        model.cleargrads()
        loss = model(x_, y_)
        loss.backward()
        optimizer.update()

        losses.append(loss.data)

        if i % 10 == 0:
            plt.plot(losses, 'b')
            plt.yscale('log')
            plt.pause(0.01)
            # plt.clf()

        if i % 100 == 0:
            serializers.save_npz('{}.model'.format(i), model)
