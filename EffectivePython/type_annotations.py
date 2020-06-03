def careful_divide(a: float, b: float) -> float:
    """Divides a by b

    Raises:
        ValueError: When the input cannot be divided

    """
    try:
        return a/b
    except ZeroDivisionError as e:
        raise ValueError('Invalid inputs')

x, y = 81, 9
try:
    result = careful_divide(x, y)
except ValueError:
    print('Invalid inputs')
else:
    print(f"Result is {result}")