# -*- coding: utf-8 -*-

try:
    import faiss
except ImportError:
    from gntp.lookup.nms import NMSLookupIndex

    class FAISSLookupIndex(NMSLookupIndex):
        pass
else:
    import numpy as np

    from gntp.lookup.base import BaseLookupIndex

    class FAISSLookupIndex(BaseLookupIndex):
        def __init__(self, data, exact=True, cpu=False, resource=None, kernel_name='rbf'):
            super().__init__()
            self.times_queried = 0
            self.exact = exact

            # creating the index
            self.index = None
            self.cpu_index_flat = None
            self.resource = resource
            self.kernel_name = kernel_name
            self.cpu = cpu

            self.build_index(data)

        def _build_approximate_index(self,
                                     data: np.ndarray):
            dimensionality = data.shape[1]
            nlist = 100 if data.shape[0] > 100 else 2

            if self.kernel_name in {'rbf'}:
                quantizer = faiss.IndexFlatL2(dimensionality)
                cpu_index_flat = faiss.IndexIVFFlat(quantizer, dimensionality, nlist, faiss.METRIC_L2)
            else:
                quantizer = faiss.IndexFlatIP(dimensionality)
                cpu_index_flat = faiss.IndexIVFFlat(quantizer, dimensionality, nlist)

            gpu_index_ivf = faiss.index_cpu_to_gpu(self.resource, 0, cpu_index_flat)
            gpu_index_ivf.train(data)
            gpu_index_ivf.add(data)
            self.index = gpu_index_ivf

        def _build_exact_index(self,
                               data: np.ndarray):
            dimensionality = data.shape[1]

            if self.kernel_name in {'rbf'}:
                self.cpu_index_flat = faiss.IndexFlatL2(dimensionality)
            else:
                self.cpu_index_flat = faiss.IndexFlatIP(dimensionality)

            if not self.cpu:
                self.index = faiss.index_cpu_to_gpu(self.resource, 0, self.cpu_index_flat)
            else:
                self.index = self.cpu_index_flat
            self.index.add(data)

        def query(self,
                  data: np.ndarray,
                  k: int = 10,
                  is_training: bool = False):
            _, neighbour_indices = self.index.search(data, k)
            self.times_queried += 1 if is_training else 0
            return neighbour_indices

        def build_index(self,
                        data: np.ndarray):
            self.times_queried = 0

            if self.exact:
                self._build_exact_index(data)
            else:
                self._build_approximate_index(data)
