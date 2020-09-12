import chainer
import chainerrl
from chainer import links as L
from chainer import functions as F
from chainer import initializers
import numpy as np

from chainerrl.action_value import DiscreteActionValue


class CNN(chainer.Chain):

    def __init__(self, num_cnn_out=256):
        super().__init__()
        self.embedding_size = num_cnn_out

        with self.init_scope():
            # L.Convolution2D(入力チャンネル数, 出力チャンネル数, フィルタサイズ, ストライド)
            self.L0=L.Convolution2D(4, 16, ksize=8, stride=4)
            self.L1=L.Convolution2D(16, 32, ksize=4, stride=2)
            self.L2=L.Convolution2D(32, 32, ksize=3, stride=1)
            self.L3=L.Linear(1568, num_cnn_out)

    def __call__(self, x, test=False):

        h0 = F.relu(self.L0(x))
        h1 = F.relu(self.L1(h0))
        h2 = F.relu(self.L2(h1))
        h3 = self.L3(h2)

        return h3

class QFunction(chainer.Chain):

    def __init__(self, num_action, num_cnn_out=256):
        super().__init__()
        self.num_action = num_action
        with self.init_scope():
            #CNNと接続
            self.hout = CNN(num_cnn_out)
            #受け取ったembeddingとQ値出力を全結合
            self.qout = L.Linear(num_cnn_out, num_action)

    def __call__(self, x, test=False):
        self.h = self.hout(x)
        #Q関数に渡す前に活性化関数に通す
        q = self.qout(F.relu(self.h))
        return DiscreteActionValue(q)

    @property
    def embedding(self):
        return self.h.array
