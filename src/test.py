import os
from units import *
import numpy as np
import ctypes

tsmlib = ctypes.cdll.LoadLibrary('tsm.so')
tsm = tsmlib.tsm

tnum_hhg = 5000
T = np.linspace(-25*fs, 25*fs, tnum_hhg)
E = 280*mv_per_cm*np.exp(-(T/10/fs)**2)*np.cos(2*np.pi*T/(5.4*fs))
X = np.zeros(tnum_hhg)

tsm(tnum_hhg,
    ctypes.c_void_p(T.ctypes.data),
    ctypes.c_void_p(E.ctypes.data),
    2,
    ctypes.c_void_p(X.ctypes.data),
    ctypes.c_double(15*ev))

for i in range(tnum_hhg):
  print(X[i])
