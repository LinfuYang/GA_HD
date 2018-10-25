from kernal import GA
from kernal.func_ND import f_GP


D = 10
opt_x = [2] * D
print(f_GP(dim=D).f(opt_x))
best_x, best_y = GA.main(max_iter=100, opt_iters=100, mun_opt=3, popula=10, func=f_GP(dim=D), transla_p=0.15,  view_plot=True)
print('*******************')
print('最优解:', end=' ')
print(best_x)
print('最优目标函数值:', best_y)


