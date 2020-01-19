# for 8.14
# t-test and model selection

import numpy as np

a = np.array([30.5, 32.2, 20.7, 20.6, 31.0, 41.0, 27.7, 26.0, 21.5, 26.0])
b = np.array([22.4, 14.5, 22.4, 19.6, 20.7, 20.4, 22.1, 19.4, 16.2, 35.0])
k = 10

var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

s = np.sqrt((var_a + var_b)/2.0)
t = (a.mean() - b.mean())/(s*np.sqrt(2/k))

print(t)
print(a.mean())
print(b.mean())


