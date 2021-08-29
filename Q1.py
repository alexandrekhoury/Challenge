List = [1,2,3,4] 
def find(x,List):
    count=0
    for i in List :
        if x==i: 
            return count
        count+=1
    return print("The value was not found.")

print(find(3,List))

import math
List = [1,28,34,48,72,102] 
def find_sorted(x,List):
    L=0
    R=len(List)-1	
    while (L<=R):
        i= math.floor((L+R)/2)
        if List[i]<x: 
            L+=1
        elif List[i]>x:
            R=i-1
        else:
            return i       
    return print("The value was not found.")

print(find_sorted(72,List))