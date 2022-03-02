from ctypes import CDLL, POINTER, byref, c_size_t
import sys

cudart = CDLL('libcudart.so')

cudaMemGetInfo = cudart.cudaMemGetInfo
cudaMemGetInfo.argtypes = [POINTER(c_size_t), POINTER(c_size_t)]

free = c_size_t()
total = c_size_t()
err = cudaMemGetInfo(byref(free), byref(total))

if (err):
    print(f"cudaMemGetInfo error is {err}")
    sys.exit(1)

print(f"Free: {free.value} bytes")
print(f"Total: {total.value} bytes")

print("Importing Numba")

from numba import cuda

print("Allocating array")

cuda.device_array(1)

print("Finished")