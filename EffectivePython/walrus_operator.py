# Walrus operator in python 3.8
fresh_fruit = {
    'apple': 10,
    'banana': 8,
    'lemon': 5,
}

def make_lemonade(count):
    print(f"Lemonade juice made using {count} lemons")


def out_of_stock():
    print("Sorry... we are out of stock...")

if count := fresh_fruit.get('lemon', 0): # Both assigns and evaluates variables in a single expression
    make_lemonade(count)
else:
    out_of_stock()