import sympy as sp

x = sp.Symbol('x')  # 定义符号变量
f = x ** 3 + x ** 2 + x - 3
x = sp.solve(f)
print(x)
