from pyflow import Tensor as T
from time import perf_counter
import numpy as np

l1 = np.random.rand(1000, 1000).tolist()
l2 = np.random.rand(1000, 1000).tolist()

t1 = T(l1)
t2 = T(l2)

a1 = np.array(l1)
a2 = np.array(l2)

start = perf_counter()
for i in range(10):
    t1.transpose()
end = perf_counter()

start2 = perf_counter()
for i in range(10):
    a1.T
end2 = perf_counter()

print(end-start)
print(end2-start2)