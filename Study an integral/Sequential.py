# Write as py cause without it time it show speedup for each loop (or k in range(N-1))

import math
import numpy as np
import matplotlib.pyplot as plt

def trapezia(a,b):
    N = 1000
    h = (b-a)/N
    S0 = (1/math.sqrt(1 + a**2) + 1/math.sqrt(1 + b**2))*h/2
    S1 = 0
    for k in range(N-1):
        x = a + k*h
        S1 += 1/math.sqrt(1 + x**2)
    return S0 + h*S1

a = 15
b = 21
S = trapezia(a,b)
# print(f'Trapezoidal rule: {S}')
