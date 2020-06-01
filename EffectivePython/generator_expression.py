filename = '/Users/senthilp/Desktop/mne_tutorial/my_file.txt'

# List comprehension
value = [len(x) for x in open(filename)]
print(value)

# Generator expression
# Generator expression avoid memory issue by producing outputs
# one at a time as iterators...
it = (len(x) for x in open(filename))
print(it)