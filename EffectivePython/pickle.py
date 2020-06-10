'''The pickle model allows us to store almost any python 
object in a file directly'''

import pickle

D = {'a': 1, 'b': 2}

''' w to create and open for text output
b for binary file
Store the pickle object in binary mode because pickler
creates and uses a bytes string object and these object
imply binary mode files'''

F = open('datafile.pkl', 'wb')
pickle.dump(D, F)
F.close()

F_open = open('datafile.pkl', 'rb')
E = pickle.load(F_open)
print(E)