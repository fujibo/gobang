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
            conv1=L.Convolution2D(in_channels=1, out_channels=20, ksize=3, stride=1),
            fc2=L.Linear(500, 1)
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.predict(x), y)

    def predict(self, x):
        'return predict value only used in this NN'
        h1 = F.tanh(self.conv1(x))
        h2 = F.tanh(self.fc2(h1))
        return h2

    def get(self, x):
        'return predict value by float'
        # a x 49
        return self.predict(Variable(x)).data
