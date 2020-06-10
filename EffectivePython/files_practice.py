
filename = 'myfile.txt'
myfile = open(filename, 'w')

# Write method doesnt automatically add new line
myfile.write('hello world files\n')
myfile.write('goodbye text file\n')
myfile.close()

myfile = open(filename)
#print(myfile.readline())
#print(myfile.readline())
#print(myfile.readline())

# read entire file into a string
print(open(filename).read())

# Scan a file line by line
for line in open(filename):
    print(line, end='')
    
    
    
    
    
