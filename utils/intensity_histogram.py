"""
https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html

Description:

Compute a 1D histogram over elements from an input array. Note that these Numba implementations 
do not cover all the options that numpy.histogram allows.
"""

import numpy as np
import numba
from numba import cuda

@numba.jit(nopython=True)
def compute_bin(x, n, xmin, xmax):
    # special case to mirror NumPy behavior for last bin
    if x == xmax:
        return n - 1 # a_max always in last bin

    # SPEEDTIP: Remove the float64 casts if you don't need to exactly reproduce NumPy
    bin = np.int32(n * (np.float64(x) - np.float64(xmin)) / (np.float64(xmax) - np.float64(xmin)))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin

@cuda.jit
def histogram(x, xmin, xmax, histogram_out):
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, x.shape[0], stride):
        # note that calling a numba.jit function from CUDA automatically
        # compiles an equivalent CUDA device function!
        bin_number = compute_bin(x[i], nbins, xmin, xmax)

        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            cuda.atomic.add(histogram_out, bin_number, 1)

@cuda.jit
def min_max(x, min_max_array):
    nelements = x.shape[0]

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Array already seeded with starting values appropriate for x's dtype
    # Not a problem if this array has already been updated
    local_min = min_max_array[0]
    local_max = min_max_array[1]

    for i in range(start, x.shape[0], stride):
        element = x[i]
        local_min = min(element, local_min)
        local_max = max(element, local_max)

    # Now combine each thread local min and max
    cuda.atomic.min(min_max_array, 0, local_min)
    cuda.atomic.max(min_max_array, 1, local_max)


def dtype_min_max(dtype):
    '''Get the min and max value for a numeric dtype'''
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    else:
        info = np.finfo(dtype)
    return info.min, info.max


@numba.jit(nopython=True)
def get_bin_edges(a, nbins, a_min, a_max):
    bin_edges = np.empty((nbins+1,), dtype=np.float64)
    delta = (a_max - a_min) / nbins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges



@numba.jit(nopython=True)
def generate_hist_feature(hist,bin_edges):
    assert len(hist) == len(bin_edges) , "dimension mismatch"
    return hist * bin_edges


def numba_gpu_histogram(a, bins):
    # Move data to GPU so we can do two operations on it
    a_gpu = cuda.to_device(a)

    ### Find min and max value in array
    dtype_min, dtype_max = dtype_min_max(a.dtype)
    # Put them in the array in reverse order so that they will be replaced by the first element in the array
    min_max_array_gpu = cuda.to_device(np.array([dtype_max, dtype_min], dtype=a.dtype))
    min_max[64, 64](a_gpu, min_max_array_gpu)
    a_min, a_max = min_max_array_gpu.copy_to_host()

    # SPEEDTIP: Skip this step if you don't need to reproduce the NumPy histogram edge array
    bin_edges = get_bin_edges(a, bins, a_min, a_max) # Doing this on CPU for now

    ### Bin the data into a histogram 
    histogram_out = cuda.to_device(np.zeros(shape=(bins,), dtype=np.int32))
    histogram[64, 64](a_gpu, a_min, a_max, histogram_out)
    
    return histogram_out.copy_to_host(), bin_edges


def hist_test():
    x = np.array([21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100])
    num_bins = 4
    hist,bin_edge = numba_gpu_histogram(x,num_bins)
    feature = generate_hist_feature(hist,bin_edge[:-1])
    print("hist",hist )
    print("bin_edge",bin_edge)
    # print("bin_edge_left",bin_edge[:-1])

    print("features",feature)
    
    # import plotly.express as px
    # df = px.data.tips()
    # fig = px.histogram(x, nbins=4)
    
    # fig.show()
    
    # import matplotlib.pyplot as plt
    # n, bins, patches = plt.hist(x=x, bins=4, color='#0504aa', \
    #                         alpha=0.7, rwidth=0.85)
    
    # print("hist_mat",n )
    # print("bin_edge_mat",bins)
    # plt.show()
    print("End hist testing")
    
    
if __name__ == "__main__":
    print("running testing function",hist_test())
