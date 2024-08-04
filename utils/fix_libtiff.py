import os
import ctypes
try:
    libtiff_path = '/usr/lib/x86_64-linux-gnu/libtiff.so.5'
    libtiff = ctypes.CDLL(libtiff_path)
except:
    print(f"Can't load {libtiff_path}") 