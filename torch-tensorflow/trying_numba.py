# Fibonacci without cache
from numba import jit
import time

@jit(nopython=True) # Didn't work here.
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

start = time.time()
print(fib(50))
end = time.time()
duration = end - start
print(duration)

"""from numba import jit

@jit(nopython=True)
def sum_list(numbers):
    result = 0
    for i in range(len(numbers)):
        result += numbers[i]
    return result

# Test the function
numbers = [1, 2, 3, 4, 5]
print(sum_list(numbers))"""