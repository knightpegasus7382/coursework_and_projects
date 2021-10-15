# For a pair of numbers to give an odd product, the pair must lack a factor of 2, which means each of the multiplied numbers must be odd.
# The function also checks for distinct pairs by using the 'set' data structure to store unique odd elements

def search_odd_product(seq):
    odd_elms = set()            # odd_elms is a set which only stores a maximum of one instance of each element, so for example, it does not store two 1's or two 3's (even if they appear in seq).
    for elm in seq:
        if elm%2==1:
            odd_elms.add(elm)   # a distinct odd number gets added to the set
    if len(odd_elms)>=2:
        return True             # returns True if the unique-element set of odd numbers is more than 1 (distinct) element long (there exists a pair of distinct odd numbers)
    return False                # if True hasn't been returned, return False

# print(search_odd_product([1,2,4,6,6,8,1]))     # Output must be False (no DISTINCT pair of odd numbers)
# print(search_odd_product([1,2,4,6,6,7,1]))     # Output must be True (distinct pair of odd numbers exists)

# Taking user input for sequence and checking for presence of a distinct pair of numbers giving odd product:
print()
n = int(input("Enter number of elements in the sequence: "))
sequence = []
print("Enter the elements:")
for i in range(n):
    elm = int(input())
    sequence.append(elm)

print()
print("It is ", search_odd_product(sequence), " that the sequence a distinct pair of numbers with odd product.")
print()