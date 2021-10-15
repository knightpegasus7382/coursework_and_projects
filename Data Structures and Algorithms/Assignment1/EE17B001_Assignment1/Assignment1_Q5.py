def calc(a1,b1,c1):
    res1 = (a1 + b1 == c1)
    res2 = (a1 == b1 - c1)
    res3 = (a1 * b1 == c1)                      # evaluating the Boolean values for whether a, b, c fit in each of the expressions
    return (res1, res2, res3)                   # returning a tuple of Boolean values

print()
print("Enter the values of a, b, and c: ")      # Taking user inputs for a, b, c
a = int(input())
b = int(input())
c = int(input())

ans1, ans2, ans3 = calc(a, b, c)                # Unpacking values in the returned tuple
print("Can be used in a + b = c: ", ans1)
print("Can be used in a = b - c: ", ans2)
print("Can be used in a * b = c: ", ans3)
