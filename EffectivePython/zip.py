names = ['senthil', 'caesar', 'kavinkido']
counts = [len(name) for name in names]

max_count = 0
longest_name = None

for name, count in zip(names, counts):
    if count > max_count:
        longest_name = name
        max_count = count

print(f"The longest name is: {longest_name}")

# zip creates a lazy generator that produces tuples
some = zip(names, counts)
print(next(some))
print(next(some))
print(next(some))