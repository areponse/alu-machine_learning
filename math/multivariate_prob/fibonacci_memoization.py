
def fib(n, mem=[0]):
    if n == 1 or n == 2:
        return 1
    if len(mem) >= n:
        return mem[n]
    result = fib(n-2, mem) + fib(n-1, mem)
    mem[n] = result
    return result
    


print(fib(5, mem))