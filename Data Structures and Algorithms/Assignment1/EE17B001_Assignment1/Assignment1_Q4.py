def count_vowels(string):
    string = string.lower()                                     # Converting the letters in the string to lower case
    count = 0
    for i in string:
        if i=="a" or i=="e" or i=="i" or i=="o" or i=="u":      
            count += 1                                          # Incrementing count whenever we come across a vowel in the string
    return count

# Checking the function on a user-input string
print()
string = input("Enter character string: ")
print("The number of vowels is", count_vowels(string))
print()
