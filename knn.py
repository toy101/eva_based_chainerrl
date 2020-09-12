import numpy as np

#legacy of the past
"""
import faiss
def knn(emb_list, target_embedding, k):
    #target_embedding = np.array(embedding).astype("float32")
    #memory = np.array(emb_list).astype("float32")
    dimension = len(emb_list[0])

    # BUILD THE INDEX
    index = faiss.IndexFlatL2(dimension)
    index.add(emb_list)
    _, i = index.search(target_embedding, k)

    i = i.tolist()

    return  i[0]


class KNN():
    def __init__(self, dimension=None, nlist=None,
                 buffer_size=None,
                probe_num=1, xp=np):

        assert dimension is not None
        assert buffer_size is not None

        #gpuあり
        if xp is not np:
            self.res = faiss.StandardGpuResources()

            #ボロノイ図を使うかどうか
            if nlist is not None:
                self.nlist = nlist
                self.index = faiss.GpuIndexIVFFlat(self.res, dimension, nlist, faiss.METRIC_L2,
                                                   faiss.GpuIndexIVFFlatConfig())
                self.probe_num = probe_num
            else:
                # メモリの確保
                # Required GPU memory  (only brute-force mode)
                # = float32 * dict_size * (key_dimensions + value_dimensions) * 1.2
                memory_size = 4 * buffer_size * dimension * 1.2
                self.res.setTempMemory(int(memory_size))

                self.index = faiss.GpuIndexFlatL2(self.res, dimension, faiss.GpuIndexFlatConfig())

        #gpuなし
        else:
            self.index = faiss.IndexFlatL2(dimension)


    def serch(self, emb_list, target_embedding, k):

        self.index.add(emb_list)
        d, i = self.index.search(target_embedding, k)

        i = i.tolist()
        self.index.reset()

        return i[0]

    def train(self, emb_list):

        self.index.train(emb_list)
        self.index.setNumProbes(self.probe_num)
"""
class ArgsortKnn():
    def __init__(self, capacity = None, dimension = None, xp = np):

        assert capacity is not None
        assert dimension is not None

        self.xp = xp

        #先に次元数×キャパのメモリを確保
        self.capacity = capacity
        self.emb_memory = self.xp.empty((0, dimension),
                                        dtype='float32')
        #self.current_memory_size = 0
        #self.current_start_index = 0

    def __len__(self):
        return self.emb_memory.shape[0]

    def add(self, emb_arr):

        self.emb_memory = self.xp.concatenate([self.emb_memory, emb_arr])

        if self._check_overflow():
            self.emb_memory = self.emb_memory[self.capacity:]

    def _check_overflow(self):
        if self.emb_memory.shape[0] > self.capacity:
            return True
        else:
            return False

    #For debug
    def head_emb(self):
        return self.emb_memory[0,0]

    def end_emb(self):
        return self.emb_memory[-1,0]


    def search(self, target_emb, k):

        l2distance_arr = self.xp.sum((self.emb_memory - target_emb)**2,
                                     axis=1)
        index_arr = l2distance_arr.argsort()[:k]

        index_list = index_arr.tolist()

        return index_list

    def search_with_weight(self, target_emb, k):

        l2distance_arr = self.xp.sum((self.emb_memory - target_emb) ** 2,
                                     axis=1)
        l2distance_top_k = self.xp.sort(l2distance_arr)[:k]
        index_arr = l2distance_arr.argsort()[:k]

        weight_arr = self._compute_weight(l2distance_top_k)
        index_list = index_arr.tolist()

        return index_list, weight_arr

    def _compute_weight(self, distance_arr):

        weight_arr = distance_arr / self.xp.sum(distance_arr)

        return weight_arr


