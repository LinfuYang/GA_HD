from kernal import GA
from kernal.func_ND import f_SW



best_x, best_y = GA.main(max_iter=100, opt_iters=100, mun_opt=3, popula=10, func=f_SW(dim=10), view_plot=True)

print('*******************')
print('最优解: x1, x2, x3')
print(best_x)
print('最优目标函数值:', -best_y)
