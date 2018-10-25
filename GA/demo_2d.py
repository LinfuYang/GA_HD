from kernal.func_2D import f_1
from kernal.GA import main

x, y = main(max_iter=500, opt_iters=1, mun_opt=2, popula=20, func=f_1(dim=2))
print(x)
print(y)
