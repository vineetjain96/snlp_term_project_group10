from collections import defaultdict
from copy import deepcopy
import string
import re
import pickle 

## Punctuation remover
translator = str.maketrans('', '', string.punctuation)
## regex for finding sloka number
regex = r"(\d+)\-(\d+)\-(\d+)" 

# ANy is the prose, Slo is the sloka
slpAny = [item.split() for item in open('slpAny.txt').read().splitlines()]
slpSlo = [item.split() for item in open('slpSlo.txt').read().splitlines()]

print(slpAny[0])

sloAnv = defaultdict(dict)
for i,item in enumerate(slpSlo):
    if item[0].isnumeric() == False:
        print (item)
        
    sloAnv[int(item[0])]['slo'] = list()
    for stuff in item[1:]:
        match = re.search(regex, stuff)
        if match is not None:
            if match.start() != 0:
                stuff2 = re.sub(regex,'',stuff)

            elif match.end()!= len(stuff):
                stuff2 = re.sub(regex,'',stuff)
        else:
            stuff2 = stuff
        
        stuff2 = stuff2.strip().translate(translator).strip()
        if len(stuff2) > 0:
            sloAnv[int(item[0])]['slo'].append(stuff2)
for item in slpAny:
    if item[0].isnumeric() == False:
        print ('anv',item)
        
    sloAnv[int(item[0])]['anv'] = [stuff.strip().translate(translator).strip() for stuff in item[1:]]

# for first line

#Poetry
print("Poetry is",sloAnv[0]['slo'])


# Prose
print("Prose is",sloAnv[0]['anv'])


pickle.dump(sloAnv, open('sloAnv.pkl', 'wb'))