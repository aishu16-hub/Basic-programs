import random

numbers = [random.randint(1, 1000) for _ in range(100)]
numbers.sort()
print(numbers)
