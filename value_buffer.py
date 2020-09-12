from abc import ABCMeta
from abc import abstractmethod
from future.utils import with_metaclass
from chainer import cuda
from chainerrl.misc.collections import RandomAccessQueue

import numpy as np

import knn

class AbstractValueBuffer(with_metaclass(ABCMeta, object)):

    @abstractmethod
    def backup(self, trajectories):
        raise NotImplementedError

    @abstractmethod
    def _trajectory_centric_planning(self, trajectories):
        raise NotImplementedError

    @abstractmethod
    def _store(self, dictionaries):
        raise NotImplementedError


class ValueBuffer(with_metaclass(ABCMeta, object)):
    """non-parametricQ値を出力するためのbuffer"""

    def __init__(self, capacity = 2000, lookup_k = 5, n_action = None,
                 key_size = 256, xp = np):
        
        self.capacity = capacity
        self.memory = RandomAccessQueue(maxlen=capacity)
        self.lookup_k = lookup_k
        self.xp = xp
        self.num_action = n_action
        self.key_size = key_size
        assert self.num_action

        self.tmp_emb_arr = self.xp.empty((0, self.key_size),
                                     dtype='float32')

        self.knn = knn.ArgsortKnn(capacity = self.capacity,
                                  dimension=key_size, xp = self.xp)

    def __len__(self):
        return len(self.memory)

    def store(self, embedding, q_np):

        # value bufferに保存する
        self._store(dict(embedding = embedding, action_value = q_np))
        #knnにembeddingを送る
        self.knn.add(embedding)

        assert len(self.knn) == len(self.memory)
        assert self.memory[0]['embedding'][0,0] == self.knn.head_emb()
        if len(self.memory) == self.capacity:
            assert self.memory[-1]['embedding'][-1,0] == self.knn.end_emb()

        # 戻り値はなし (必要ならつける)
        return


    def _store(self, dictionaries):
        # 蓄える(容量いっぱいのときなどの処理は場合分け)
        self.memory.append(dictionaries)
        while self.capacity is not None and \
            len(self.memory) > self.capacity:
            self.memory.popleft()


    def compute_q(self, embedding):

        """
        if len(self.memory) < self.lookup_k:
            k = len(self.memory)
        else:
            k = self.lookup_k
        """

        index_list = self.knn.search(embedding, self.lookup_k)

        tmp_vbuf = self.xp.asarray([self.memory[i]['action_value'] for i in index_list], dtype=self.xp.float32)

        q_np = self.xp.average(tmp_vbuf, axis=0)

        return q_np

class WeightedValueBuffer(ValueBuffer):
    def __init__(self, capacity=2000, lookup_k=5, n_action=None,
                 key_size=256, xp=np):

        super().__init__(capacity=capacity, lookup_k=lookup_k,
                         n_action=n_action, key_size=key_size, xp=xp)

    def compute_q(self, embedding):

        if len(self.memory) < self.lookup_k:
            k = len(self.memory)
        else:
            k = self.lookup_k

        index_list, weight_arr  = self.knn.search_with_weight(embedding, k)

        tmp_vbuf = self.xp.asarray([self.memory[i]['action_value'] for i in index_list], dtype=self.xp.float32)

        weight_arr = weight_arr.reshape(-1,1)
        tmp_vbuf = tmp_vbuf * weight_arr

        q_np = self.xp.sum(tmp_vbuf, axis=0)

        return q_np



