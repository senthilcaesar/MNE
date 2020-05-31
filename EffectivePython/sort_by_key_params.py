'''
The key parameter of the sort method can be used to supply a
helper function that returns the value to use for sorting in 
place of each item from the list...
'''
class Tool:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    def __repr__(self):
        return f"Tool({self.name}, {self.weight})"

tools = [
    Tool('level', 3.5),
    Tool('hammer', 1.25),
    Tool('screwdriver', 0.5),
    Tool('chisel', 0.25),
]

print(f"Unsorted: {repr(tools)}")
tools.sort(key=lambda x: x.name)
print(f"\nSorted by name: {repr(tools)}")
tools.sort(key=lambda x: x.weight)
print(f"\nSorted by weight: {repr(tools)}")

# Sort first by weight and then by name
power_tools = [
    Tool('drill', 4),
    Tool('circular saw', 5),
    Tool('jackhammer', 40),
    Tool('sander', 4),
]

power_tools.sort(key=lambda x: (x.weight, x.name))
print(f"\n{power_tools}")