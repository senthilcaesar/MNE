# **kwargs catch-all parameter

def my_func(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")

my_func(goose='gosling', kangaroo='joey')