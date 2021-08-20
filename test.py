
def a(x,y):
    return [1*x,1*y]
def b(x,y):
    return [2*x,2*y]

func_dic={'a':a,'b':b}
def func(m,x,y):
    return func_dic.get(m)(x,y)

print(func('b',2,4))