# Unpacking
snacks = [('bacon', 350), ('donut', 240), ('muffin', 190)]
'''
Reduce visual noise and increase code clarity by using unpacking
to avoid explicity indexing into sequences
'''
for i, (name, calories) in enumerate(snacks, 1):  # Here "1" represents that starting index for variable "i"
    print(f"#{i}: {name} has {calories} calories")

# Prefer enumerate over range
flavor_list = ['vanilla', 'chocolate', 'pistachio', 'hazelnut']
'''
You want to iterate over the list and also know the index
of the ccurrent item in the list
'''
for i, flavor in enumerate(flavor_list): # enumarate wraps any iterator with a lazy generator
    print(f"{i+1}: {flavor}")

# Prefer catch-all unpacking over slicing
car_ages = [0, 9, 4, 8, 7, 20, 19, 1, 6, 15]
car_ages_descending = sorted(car_ages, reverse=True)
oldest, second_oldest, *others = car_ages_descending
print(oldest, second_oldest, others)
oldest, *others, youngest = car_ages_descending
print(oldest, youngest, others)
*others, second_youngest, youngest = car_ages_descending
print(youngest, second_youngest, others)