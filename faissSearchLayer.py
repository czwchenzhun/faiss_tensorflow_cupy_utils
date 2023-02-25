import warnings
import tensorflow as tf
from tensorflow.keras import layers

dependencies_ok = True
try:
    import faiss
    import cupy as cp
except:
    dependencies_ok = False
    warnings.warn("The faiss search backend is not available because faiss or cupy is not installed!")
if not dependencies_ok:
    FaissSearchLayer = None
else:
    from faissCupyUtil import cupy_knn_gpu


    class FaissSearchLayer(layers.Layer):
        def __init__(self, xb, k, metric=faiss.METRIC_L2, **kwargs):
            super(FaissSearchLayer, self).__init__(**kwargs)
            self.xb = tf.convert_to_tensor(xb)
            self.k = k
            self.metric = metric
            self.usegpu = 'gpu' in self.xb.device.lower()
            if self.usegpu:
                # Using dlpack for type conversions between frameworks, there is no extra overhead
                cap = tf.experimental.dlpack.to_dlpack(self.xb)
                self.xb = cp.fromDlpack(cap)
            else:

                self.index = faiss.IndexFlatL2(self.xb.shape[1])
                self.xb_ = self.xb.numpy()
                self.index.add(self.xb_)

        @tf.function
        def call(self, xq):
            def func(q):
                if self.usegpu:
                    q = cp.array(q.numpy())
                    res = faiss.StandardGpuResources()
                    D, I = cupy_knn_gpu(res, q, self.xb, self.k, metric=self.metric)
                    I = tf.experimental.dlpack.from_dlpack(I.toDlpack())
                    D = tf.experimental.dlpack.from_dlpack(D.toDlpack())
                else:
                    D, I = self.index.search(q.numpy(), self.k)
                    I = tf.convert_to_tensor(I)
                    D = tf.convert_to_tensor(D)
                return D, I

            # Use py_function to make the FaissSearchLayer support the TensorFlow graph mode
            D, I = tf.py_function(func, [xq], Tout=[tf.float32, tf.int64])
            I = tf.ensure_shape(I, [None, self.k])
            D = tf.ensure_shape(D, [None, self.k])
            return D, I

if __name__ == '__main__':
    import numpy as np

    np.random.seed(1234)

    d = 10  # dimension
    nb = 500  # database size
    nq = 1000  # nb of queries
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000

    with tf.device('/cpu:0'):
        # Convert to tensor on the specified device
        xb = tf.convert_to_tensor(xb)
        xq = tf.convert_to_tensor(xq)

        # Execute faiss search
        D, I = FaissSearchLayer(xb, 5)(xq)

    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries
    print(I.device)
