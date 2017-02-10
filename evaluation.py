import chainer
from chainer import Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np

# http://qiita.com/ashitani/items/1dc0a54da218ec224ad8
class MyChain(Chain):
    """docstring for MyChain"""
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(49, 64),
            l2=L.Linear(64, 1),
        )
    def __call__(self, x, y):
        return F.mean_squared_error(self.predict(x), y)

    def predict(self, x):
        'return predict value only used in this NN'
        h1 = F.tanh(self.l1(x))
        h2 = F.tanh(self.l2(h1))
        # h3 = F.tanh(self.l3(h2))
        return h2

    def get(self, x):
        'return predict value by float'
        # a x 49
        # x = x.astype(np.float32)
        return self.predict(Variable(x)).data
