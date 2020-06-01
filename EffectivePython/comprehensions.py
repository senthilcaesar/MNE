'''
List comprehensions are clearer than the map and filter
buit-in functions because they don't require
lambda expressions...
'''
a = [1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10]

squares = [x**2 for x in a]
print(squares)

even_squares = [x**2 for x in a if x%2 == 0]
print(even_squares)

even_square_dict = {x:x**2 for x in a if x%2 == 0}
print(even_square_dict)

# List comprehension two subexpresssion
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [x for row in matrix for x in row]
print(flat)

# squaring cells in matrix
cell_square = [[x**2 for x in row] for row in matrix]
print(cell_square)

# mutitple conditions
# Comprehensions with more than 2 control subexpressions
# are diffuclt to read and should be avoided
c = [x for x in a if x > 4 and x % 2 == 0]
print(c)

# Generator are produced by functions that use yield expressions
def index_word_iter(text):
    if text:
        yield 0
    for index, letter in enumerate(text):
        if letter == ' ':
            yield index + 1

address = 'Foir score and seven years ago...'
''' When called, a generator function does not actually run
but instead immediately returns an iterator...
'''
it = index_word_iter(address)
print(next(it))
print(next(it))
print(next(it))
# Convert the iterator returned by generator to a list
result = list(index_word_iter(address))
print(result)