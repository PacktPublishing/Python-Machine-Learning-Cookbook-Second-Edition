import numpy as np
from functools import reduce

def add3(input_array):
    return map(lambda x: x+3, input_array)

def mul2(input_array):
    return map(lambda x: x*2, input_array)

def sub5(input_array):
    return map(lambda x: x-5, input_array)

def function_composer(*args):
    return reduce(lambda f, g: lambda x: f(g(x)), args)

if __name__=='__main__':
    arr = np.array([2,5,4,7])

    print("Operation: add3(mul2(sub5(arr)))")
    
    arr1 = add3(arr)
    arr2 = mul2(arr1)
    arr3 = sub5(arr2)
    print("Output using the lengthy way:", list(arr3))

    func_composed = function_composer(sub5, mul2, add3)
    print("Output using function composition:", list(func_composed(arr))) 

    print("Operation: sub5(add3(mul2(sub5(mul2(arr)))))\nOutput:", \
            list(function_composer(mul2, sub5, mul2, add3, sub5)(arr)))