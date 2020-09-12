from chainerrl import replay_buffer

import numpy as np
import knn

class EVAReplayBuffer(replay_buffer.ReplayBuffer):
    def __init__(self, capacity = 500000,
                 num_steps = 1, k_num = 10, key_size = 256, xp = np):
        super().__init__(capacity=capacity, num_steps=num_steps)

        self.follow_range = 50
        self.key_size = key_size
        self.k_num = k_num
        self.xp = xp
        self.current_embeddings = self.xp.empty((0, self.key_size), dtype='float32')
        #self.current_embeddings = []
        self.knn = knn.ArgsortKnn(capacity = capacity,
                                  dimension=key_size, xp = xp)

    def lookup(self, embedding):

        # k近傍のindexを取得
        # 例: [経験1のindex, 経験3のindex, ...]

        self.update_knn_embeddings()
        #emb配列を初期化
        self.current_embeddings = self.xp.empty((0, self.key_size), dtype='float32')

        start_index = self.knn.search(embedding, self.k_num)
        # 対応するindexの経験と, そのサブシーケンス(軌跡を長さTで切り取ったもの)を取り出す
        # 例: [[経験1からの軌跡], [経験3からの軌跡], ...]

        trajectory_list = []
        for e in start_index:
            trajectory = []
            for sub_sequence in range(self.follow_range):
                step = self.memory[e + sub_sequence]
                trajectory.append(step[0])
                if step[0]["is_state_terminal"]:
                    break
                if (e + sub_sequence) == (len(self.memory) - 1):
                    break
            trajectory_list.append(trajectory)

        # 軌跡の情報を返す
        return trajectory_list

    def add(self, state, action, reward, embedding,
            next_state=None, next_action=None,
               is_state_terminal=False, **kwargs):

        experience = dict(state=state, action=action, reward=reward,
                          next_state=next_state,
                          next_action=next_action,
                          embedding = embedding,
                          is_state_terminal=is_state_terminal,
                          **kwargs)
        self.last_n_transitions.append(experience)
        if is_state_terminal:
            while self.last_n_transitions:
                self.memory.append(list(self.last_n_transitions))
                #self.current_embeddings += [m['embedding'][0] for m in self.last_n_transitions]
                for m in self.last_n_transitions:
                    self.current_embeddings = self.xp.concatenate([self.current_embeddings, m['embedding']])
                del self.last_n_transitions[0]
            assert len(self.last_n_transitions) == 0
        else:
            if len(self.last_n_transitions) == self.num_steps:
                self.memory.append(list(self.last_n_transitions))
                #self.current_embeddings += [m['embedding'][0] for m in self.last_n_transitions]
                for m in self.last_n_transitions:
                    self.current_embeddings = self.xp.concatenate([self.current_embeddings, m['embedding']])
        """
        if is_state_terminal:
            #メモリに必要な分だけ追加してくれる
            #EpisodicReplayで使う
            self.stop_current_episode()
        """

        #self.emb_arr = self.xp.concatenate([self.emb_arr, embedding])

    def stop_current_episode(self):
        # if n-step transition hist is not full, add transition;
        # if n-step hist is indeed full, transition has already been added;
        if 0 < len(self.last_n_transitions) < self.num_steps:
            self.memory.append(list(self.last_n_transitions))
            for m in self.last_n_transitions:
                self.current_embeddings = self.xp.concatenate([self.current_embeddings, m['embedding']])
        # avoid duplicate entry
        if 0 < len(self.last_n_transitions) <= self.num_steps:
            del self.last_n_transitions[0]
        while self.last_n_transitions:
            self.memory.append(list(self.last_n_transitions))
            del self.last_n_transitions[0]
        assert len(self.last_n_transitions) == 0

    def update_knn_embeddings(self):
        if len(self.current_embeddings) > 0:
            self.knn.add(self.current_embeddings)
        assert self.capacity >= len(self.knn)





