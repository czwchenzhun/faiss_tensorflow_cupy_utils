"""
Interoperability functions for Cupy and Faiss: Importing this will allow
Cupy ndarray to be used as arguments to Faiss indexes and
other functions. 
"""

import faiss
import contextlib
import cupy as cp
import numpy as np


def swig_ptr_from_UInt8CupyArray(x):
    """ gets a Faiss SWIG pointer from a cupy ndarray (on CPU or GPU) """
    assert x._c_contiguous
    assert x.dtype == cp.uint8
    return faiss.cast_integer_to_uint8_ptr(x.data.ptr)


def swig_ptr_from_HalfCupyArray(x):
    """ gets a Faiss SWIG pointer from a cupy ndarray (on CPU or GPU) """
    assert x._c_contiguous
    assert x.dtype == cp.float16
    # no canonical half type in C/C++
    return faiss.cast_integer_to_void_ptr(x.data.ptr)


def swig_ptr_from_FloatCupyArray(x):
    """ gets a Faiss SWIG pointer from a cupy ndarray (on CPU or GPU) """
    assert x._c_contiguous
    assert x.dtype == cp.float32
    return faiss.cast_integer_to_float_ptr(x.data.ptr)


def swig_ptr_from_IntCupyArray(x):
    """ gets a Faiss SWIG pointer from a cupy ndarray (on CPU or GPU) """
    assert x._c_contiguous
    assert x.dtype == cp.int32, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_int_ptr(x.data.ptr)


def swig_ptr_from_IndicesCupyArray(x):
    """ gets a Faiss SWIG pointer from a cupy ndarray (on CPU or GPU) """
    assert x._c_contiguous
    assert x.dtype == cp.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_idx_t_ptr(x.data.ptr)


@contextlib.contextmanager
def using_stream(res, cuda_stream=None):
    """ Creates a scoping object to make Faiss GPU use the same stream
        as pytorch, based on torch.cuda.current_stream().
        Or, a specific pytorch stream can be passed in as a second
        argument, in which case we will use that stream.
    """

    if cuda_stream is None:
        cuda_stream = cp.cuda.get_current_stream()

    # This is the cudaStream_t that we wish to use
    cuda_stream_s = faiss.cast_integer_to_cudastream_t(cuda_stream.ptr)

    # So we can revert GpuResources stream state upon exit
    prior_dev = cp.cuda.get_device_id()
    prior_stream = res.getDefaultStream(prior_dev)

    res.setDefaultStream(prior_dev, cuda_stream_s)

    # Do the user work
    try:
        yield
    finally:
        res.setDefaultStream(prior_dev, prior_stream)


def cupy_knn_gpu(res, xq, xb, k, D=None, I=None, metric=faiss.METRIC_L2, device=-1):
    if type(xb) is np.ndarray:
        # Forward to faiss __init__.py base method
        return faiss.knn_gpu_numpy(res, xq, xb, k, D, I, metric, device)

    nb, d = xb.shape
    if xb._c_contiguous:
        xb_row_major = True
    elif xb.transpose()._c_contiguous:
        xb = xb.transpose()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    if xb.dtype == cp.float32:
        xb_type = faiss.DistanceDataType_F32
        xb_ptr = swig_ptr_from_FloatCupyArray(xb)
    elif xb.dtype == cp.float16:
        xb_type = faiss.DistanceDataType_F16
        xb_ptr = swig_ptr_from_HalfCupyArray(xb)
    else:
        raise TypeError('xb must be f32 or f16')

    nq, d2 = xq.shape
    assert d2 == d
    if xq._c_contiguous:
        xq_row_major = True
    elif xq.transpose()._c_contiguous:
        xq = xq.transpose()
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    if xq.dtype == cp.float32:
        xq_type = faiss.DistanceDataType_F32
        xq_ptr = swig_ptr_from_FloatCupyArray(xq)
    elif xq.dtype == cp.float16:
        xq_type = faiss.DistanceDataType_F16
        xq_ptr = swig_ptr_from_HalfCupyArray(xq)
    else:
        raise TypeError('xq must be f32 or f16')

    if D is None:
        with cp.cuda.Device(xb.device):
            D = cp.empty((nq, k), dtype=cp.float32)
    else:
        assert D.shape == (nq, k)
        # interface takes void*, we need to check this
        assert (D.dtype == cp.float32)

    if I is None:
        with cp.cuda.Device(xb.device):
            I = cp.empty((nq, k), dtype=cp.int64)
    else:
        assert I.shape == (nq, k)

    if I.dtype == cp.int64:
        I_type = faiss.IndicesDataType_I64
        I_ptr = swig_ptr_from_IndicesCupyArray(I)
    elif I.dtype == I.dtype == cp.int32:
        I_type = faiss.IndicesDataType_I32
        I_ptr = swig_ptr_from_IntCupyArray(I)
    else:
        raise TypeError('I must be i64 or i32')

    D_ptr = swig_ptr_from_FloatCupyArray(D)

    args = faiss.GpuDistanceParams()
    args.metric = metric
    args.k = k
    args.dims = d
    args.vectors = xb_ptr
    args.vectorsRowMajor = xb_row_major
    args.vectorType = xb_type
    args.numVectors = nb
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.queryType = xq_type
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr
    args.outIndicesType = I_type
    args.device = device

    with using_stream(res):
        faiss.bfKnn(res, args)
    return D, I


if __name__ == '__main__':
    import numpy as np

    np.random.seed(1234)

    d = 10  # dimension 
    nb = 500  # database size
    nq = 4000000  # nb of queries
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 10000
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 10000

    xb = cp.array(xb)  # to cupy ndarray
    xq = cp.array(xq)

    k = 4  # number of nearest neighbors
    res = faiss.StandardGpuResources()
    D, I = cupy_knn_gpu(res, xq, xb, k)

    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries
