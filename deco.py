
# def deco(*args, **kwargs):
#     def wrapper(func):
#         def inner_func(a, b):
#             result = func(a,b)
#             return result+5+args[0]+kwargs["f"]
#         return inner_func
#     return wrapper

# @deco(3, f=10)
# def add_5(a, b):
#     return a+b

# print(add_5(1,2))

import asyncio

async def func():
    print("hello")
    await asyncio.sleep(2)
    print("world")