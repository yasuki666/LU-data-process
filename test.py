import random
import numpy as np
import time
n = 15000
time_1 = time.time()
a = []
for i in range(n):
    a.append(random.randint(0,1))
time_2 = time.time()
b = []
for i in range(n):
    b.append((n- 2*i + np.sum(a[:i]) + (n-i)*a[n-i-1]) / n)
time_3 = time.time()
c= []
sum = 0
for i in range(n):

    c.append((n - 2 * i + sum + (n - i) * a[n - i - 1]) / n)
    sum += a[i]

for i in range(n):
    if b[i] != c[i]:
        print('no')
print(time_3-time_2)
print(time_2-time_1)

