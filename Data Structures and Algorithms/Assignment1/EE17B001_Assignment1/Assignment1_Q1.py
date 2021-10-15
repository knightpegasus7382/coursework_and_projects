
# (a)

def sum_squares(n):
    assert n % 1 == 0               # Checking that n is an integer (returning AssertionError otherwise)
    assert n - 1 > 0                # Checking if the largest integer smaller than n is also positive, as required
    add = 0
    for i in range(1,n):
        add += i**2
    return add

print("Enter the value of n for sum of squares of positive integers less than n:")
num = int(input())
print("Answer = ", sum_squares(num))


# (b)

def sum_odd_squares(n):
    assert n % 1 == 0               # Checking that n is an integer (returning AssertionError otherwise)
    assert n - 1 > 0                # Checking if the largest integer smaller than n is also positive, as required
    add = 0
    for i in range(1,n):
        if i%2 == 1:                # Add square of the number only if it's odd
            add += i**2
    return add

print("Enter the value of n for sum of squares of ODD positive integers less than n:")
num = int(input())
print("Answer = ", sum_odd_squares(num))
