import re

input_file = 'TextFile.txt' 

data = {}

data['Twain'] = open(input_file,'r').read()

print(data['Twain'])

#Conversion to lower case
for k in data:
    data[k] = data[k].lower()
    
#Removing punctuation
for k in data:
    data[k] = re.sub(r'[-./?!,":;()\']',' ',data[k])
    
#Removing number
for k in data:
    data[k] = re.sub('[-|0-9]',' ',data[k])
    
#remove extra whitespace
for k in data:
    data[k] = re.sub(' +',' ',data[k])    

print('###########################')
print(data['Twain'])


with open('Twain.csv', 'w') as f:
    for key in data.keys():
        f.write("%s,%s\n"%(key,data[key]))
        
f.close()