from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("C_Code/Cython_Code/mouse_ar.pyx")
)

setup(
    ext_modules=cythonize("C_Code/Cython_Code/_pyautogui_osx.pyx")
)

setup(
    ext_modules=cythonize("C_Code/Cython_Code/moveMouseOptimiced.pyx")
)