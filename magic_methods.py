"""
类中魔法方法：
    （1）构造方法
    （2）__call__
    （3）__len__


"""


class Add:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def sum(self):
        return self.x + self.y

    def __call__(self, x, y):
        return x+y

# __call__的调用
add_method = Add(10, 20)
print(add_method(1,1))
print(add_method.__call__(1,1))