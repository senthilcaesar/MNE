class Car:
    
    def __init__(self, color, mileage):
        self.color = color
        self.mileage = mileage
        
    def __str__(self):
        return f'a {self.color} car'
    
mycar = Car('red', 37281)

''' Printing the object resulted in the string
returned by the __str__ method we added '''

print(mycar)