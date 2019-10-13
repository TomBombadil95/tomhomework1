#!/usr/bin/env python
# coding: utf-8

# # Say Hello World

# In[ ]:


print("Hello, World!")


# # Python If-Else

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())




if n % 2 == 0 and n <= 5:
        print("Not Weird")
elif n % 2== 0 and n > 5 and n <= 20:
        print ("Weird")
elif n %2==0 and n > 20:
        print ("Not Weird")
else:
        print("Weird")


# # Arithmetic Operators

# In[ ]:


if __name__ == '__main__':
    a = int(input())
    b = int(input())

print(a+b)
print(a-b)
print(a*b)


# # Python Division    

# In[ ]:


if __name__ == '__main__':
    a = int(input())
    b = int(input())

print(a//b)
print(a/b)


# # Loops

# In[ ]:


if __name__ == '__main__':
    n = int(input())


for i in range(n):
    print(i*i)


# # Write a function

# In[ ]:


def is_leap(year):
    leap = False
    return year % 4 == 0 and (year % 400 == 0 or year % 100 != 0)


year = int(input())
print(is_leap(year))


# # Print Function
# 

# In[ ]:


if __name__ == '__main__':
    n = int(input())

for i in range(1,n+1):
    print(i, end = "")


# # List Comprehension

# In[ ]:


if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

print([[i,j,k] for i in range(x+1) for j in range(y+1) 
       for k in range(z+1) if (i+j+k !=n)])


# # Find the runner up score
# 

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    arr =(map(int, input().split()))


arr = list(arr)
m = max(arr)
while max(arr) == m:
    arr.remove(m)
print(max(arr))


# # Finding the percentage

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        scores=sum(scores)/3
        student_marks[name] = scores
    query_name = input()    
    print('%.2f' % student_marks[query_name])


# # Lists

# In[ ]:


if __name__ == '__main__':
    N = int(input())
    l = []
for i in range(N):
    cosafare = input().split()
    if cosafare[0] == "insert":
        l.insert(int(cosafare[1]),int(cosafare[2]))
    if cosafare[0] == "append":
        l.append(int(cosafare[1]))
    if cosafare[0] == "remove": 
        l.remove(int(cosafare[1]))
    if cosafare[0] == "pop":
        l.pop()
    if cosafare[0] == "sort":
        l.sort()
    if cosafare[0] == "reverse":
        l.reverse()
    if cosafare[0] == "print":
        print(l)


# # Tuples

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())


# # sWAP cASE

# In[ ]:


def swap_case(s):
    r = ""
    for i in s:
        if i.isupper() == True:
            r+=(i.lower())
        else:
            r+=(i.upper())
    return r
         

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


# # String split and join
# 

# In[ ]:


def split_and_join(line):
  line = line.split(" ")
  line = "-".join(line)
  return line
  

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# # What's your name
# 

# In[ ]:


def print_full_name(a, b):
    print("Hello {} {}! You just delved into python." .format(a,b))

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


# # Mutations

# In[ ]:


def mutate_string(string, position, character):
    s =list(string) 
    i = position
    c = character
    s[i] = c    
    return ''.join(s)
    
if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


# # Find a string

# In[ ]:


def count_substring(string, sub_string):
    conta=0
    i = 0
    while i<len(string):
        if string.find(sub_string, i)>=0:
            i = string.find(sub_string,i)+1
            conta+=1
        else: 
            break
    return conta


if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)


# # Text alignment

# In[ ]:


#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# # Text Wrap

# In[ ]:


import textwrap

def wrap(string, max_width):
    for i in range(0,len(string), max_width):
        "\n".join([string[i:i+max_width]])
    return string




def wrap(string, max_width):
    return "\n".join([string[i:i+max_width] for i in range(0, len(string), max_width)])

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


# # Designer Door Mat

# In[ ]:


n, m = map(int,input().split())
pattern = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
print('\n'.join(pattern + ['WELCOME'.center(m, '-')] + pattern[::-1]))


# # String Formatting

# In[ ]:


def print_formatted(number):
    width = len(bin(number)[2:])
    for i in range(1,number+1):
        print(str(i).rjust(width,' '),end=" ")
        print(oct(i)[2:].rjust(width,' '),end=" ")
        print(((hex(i)[2:]).upper()).rjust(width,' '),end=" ")
        print(bin(i)[2:].rjust(width,' '),end=" ")
        print("")

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


# # collection.Counter()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
x = int(input())
import collections
from collections import Counter
sizes = collections.Counter(map(int, input().split()))
c = int(input())
cassa = 0
for i in range(c):
    s, p = map(int, input().split())
    if sizes[s]:
        cassa += p
        sizes[s] -= 1
print (cassa)


# # Introduction to Sets

# In[ ]:


def average(array):
    return (sum(set(array))/len(set(array)))


if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)


# # DefaultDict Tutorial

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import collections
from collections import defaultdict
d = defaultdict(list)
l = []
n , m = map(int, input().split())
for i in range(n):
    d[input()].append(i+1)
for j in range(m):
    l+=[input()]
for k in l:
    if k in d:
        print (" ".join(map(str,d[k])))
    else:
        print (-1)



# # Calendar Module

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
date = input().split()
m = int(date[0])
d = int(date[1])
y = int(date[2])
giorno = (calendar.weekday(y, m, d)) 
if giorno == 0:
    print("MONDAY")
elif giorno == 1:
    print("TUESDAY")
elif giorno == 2:
    print("WEDNESDAY")
elif giorno ==3:
    print("THURSDAY")
elif giorno ==4:
    print("FRIDAY")
elif giorno == 5:
    print("SATURDAY")
elif giorno ==6:
    print("SUNDAY")


# # Exceptions

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
t = int(input())
for i in range(t):
    a, b = input().split()
    try:
        print(int(a)//int(b))
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as v:
        print("Error Code:", v)


# # collections.namedtuple()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import collections
from collections import namedtuple
n = int(input())
col = input().split()
tot = 0
for i in range(n):
    stud = namedtuple("student", col)
    col1, col2, col3, col4 = input().split()
    student = stud(col1,col2,col3,col4)
    tot += int(student.MARKS)
print(tot/n)


# # collections. OrderedDict()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import collections
from collections import OrderedDict

d = OrderedDict()
n = int(input())

for i in range(n):
    m = input().split() 
    item, price = ' '.join(m[:-1]), int(m[-1])
    d[item] = d.get(item, 0) + int(price)
for j, k in d.items():
    print(j,k)


# # Symmetric Difference

# In[ ]:


n1,m1,n2,m2 = (int(input()),input().split(),int(input()),input().split())

s1=set(m1)
s2=set(m2)

diff1=s2.difference(s1)
diff2=s1.difference(s2)

r=diff1.union(diff2)

print ('\n'.join(sorted(r, key=int)))


# # Set.add()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
s = set()

for i in range(n):
    s.add(input())
print (len(s))


# # Set. discard(), remove(), pop()

# In[ ]:


n = int(input())
s = set(map(int, input().split()))

N = int(input())

for i in range(N):
    c = input().split()
    if c[0] == "pop":
        s.pop()
    elif c[0] == "remove":
        s.remove(int(c[1]))
    elif c [0] == "discard":
        s.discard(int(c[1]))
print (sum(s))


# # collections.deque()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import collections
from collections import deque
n = int(input())
d = deque()

for i in range(n):
    cosafare = list(input().split())
    if cosafare[0] == "append":
        d.append(cosafare[1])
    if cosafare[0]== "appendleft":
        d.appendleft(cosafare[1])
    if cosafare[0] == "pop":
        d.pop()
    if cosafare[0] == "popleft":
        d.popleft()
for elem in d:
    print (elem, end = " ")


# # Set.union() Operation

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
s1 = set(map(int, input().split()))
m = int(input())
s2 = set(map(int, input().split()))

s3 = s1|s2
print (len(s3))


# # Piling up

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
for t in range(int(input())):
    input()
    lenght = list(map(int, input().split()))
    l = len(lenght)
    i = 0
    while i < l - 1 and lenght[i] >= lenght[i+1]:
        i += 1
    while i < l - 1 and lenght[i] <= lenght[i+1]:
        i += 1
    print ("Yes" if i == l - 1 else "No")


# # Set.intersection() Operation

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
s1 = set(map(int, input().split()))
m =int(input())
s2 = set(map(int, input().split()))

s3 = s1&s2
print (len(s3))


# # Set. difference() Operation

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
s1 = set(map(int, input().split()))
m =int(input())
s2 = set(map(int, input().split()))

s3 = s1 - s2
print(len(s3))


# # Set. symmetric_difference() Operation

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
s1 = set(map(int, input().split()))
m =int(input())
s2 = set(map(int, input().split()))

s3 = s1^s2

print(len(s3))


# # Set mutations

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
s1 = set(map(int, input().split()))
others = int(input())

for i in range(others):
    c = input().split()
    s = set(map(int, input().split()))
    if c[0] == "intersection_update":
        s1.intersection_update(s)
    if c[0] == "update":
        s1.update(s)
    if c[0] == "symmetric_difference_update":
        s1.symmetric_difference_update(s)
    if c[0] == "difference_update":
        s1.difference_update(s)
print (sum(s1))


# # Captain's room

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
l = list(map(int, input().split()))
s = set(l)

diff = sum(s)*n - sum(l)

print (diff//(n-1))


# # Check subset
# 

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
T = int(input())
for i in range(T):
    a = int(input())
    A = set(map(int, input().split()))
    b = int(input())
    B = set(map(int, input().split()))
    if A.issubset(B):
        print (True)
    else:
        print(False)


# # Check strict superset

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
A = set(map(int, input().split()))
n = int(input())
for i in range(n):
    s = set(map(int, input().split()))
    if not A.issuperset(s):
        print("False")
        break
else:
    print("True")


# # Zipped

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n,x = map(int,input().split())
l=[]
for i in range(x):
    l.append(map(float,input().split()))
l = list(zip(*l))
for i in range(n):
    print(sum(l[i])/x)


# # Input()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
x, k = map(int, input().split())

print(eval(input()) == k)


# # Python evaluation

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT

eval(input())


# # Detect Floating Point Number

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
T = int(input())

for i in range(T):
    try:
        if float(input()):
            print (True)
        else:
            print (False)
    except:
        print( False)


# # Map and Lambda function

# In[ ]:


cube = lambda x: x*x*x # complete the lambda function 

def fibonacci(n):
    a,b = 0,1
    for i in range(n):
        yield a
        a,b = b,a+b
        
    # return a list of fibonacci numbers


# # re.split()

# In[ ]:


regex_pattern = r"[,.]"	# Do not delete 'r'.


# # group(), groups() , groupdict()

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
S = input()
m = re.search(r'([a-z A-Z 0-9])\1', S)
if m:
    print(m.group(1))
else: 
    print(-1)


# # Validating Roman Numerals

# In[ ]:


regex_pattern = r"^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"	# Do not delete 'r'


# # Validating Phone Numbers

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
for i in range(n):
    if re.match(r"^[789]{1}\d{9}$", input()):
        print("YES")
    else:
        print("NO")


# # Standardize Mobile Number with Decorators

# In[ ]:


def wrapper(f):
    
    def fun(l):
            f(("+91 "+c[-10:-5]+" "+c[-5:]) for c in l)
        # complete the function
    return fun


# # String validators

# In[ ]:


if __name__ == '__main__':
    s = input()
    r = ["False", "False", "False", "False", "False"]
for i in s:
    if i.isalnum():
        r[0] = "True"
    if i.isalpha():
        r[1] = "True"
    if i.isdigit():
        r[2] = "True"
    if i.islower():
        r[3] = "True"
    if i.isupper():
        r[4] = "True"
print(*r, sep="\n")


# # Capitalize

# In[1]:


# Complete the solve function below.
def solve(s):
    nome = s.split(" ")
    maius = [i.capitalize() for i in nome]
    return (" ".join(maius))


# # Incorrect Regex

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
t = int(input())

for i in range(t):
    try:
        re.compile(input())
        print (True)
    except re.error:
        print (False)


# # Any or All

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
m = input().split()
print((all(int(i)>0 for i in m))and(any(i==i[::-1] for i in m)))


# # Arrays

# In[ ]:


def arrays(arr):
    arr2 = arr[::-1]
    return numpy.array(arr2, float)
    # complete this function
    # use numpy.array


# In[ ]:





# # Shape and Reshape

# In[ ]:


import numpy
M = list(map(int, input().split()))
print(numpy.reshape(M, (3,3)))


# # Transpose and Flatten

# In[ ]:


import numpy
n, m = map(int, input().split())
arr = numpy.array([input().split() for i in range(n)], int)
print(numpy.transpose(arr))
print(arr.flatten())


# # Concatenate

# In[ ]:


import numpy
n,m,p = map(int, (input().split()))
arr = numpy.array([input().split() for i in range(n)], int)
arr2 = numpy.array([input().split() for i in range(m)], int)

print (numpy.concatenate((arr, arr2), axis=0))


# # Zeros and Ones

# In[ ]:


import numpy
dim = list(map(int, input().split()))
print (numpy.zeros((dim), dtype = numpy.int))
print (numpy.ones((dim), dtype = numpy.int))


# # Eye and Identity

# In[ ]:


import numpy
n, m = map(int, input().split())
numpy.set_printoptions(sign=' ') #risolve il bug dei vari test cases
print(numpy.eye(n, m, k=0))


# # ArrayMathematics

# In[ ]:


import numpy
n,m = map(int, input().split())
a = numpy.array([input().split() for i in range(n)], dtype=int)
b = numpy.array([input().split() for i in range(n)], dtype=int)
print (a+b)
print (a-b)
print (a*b)
print (a//b)
print (a%b)
print (a**b)


# # Floor, Ceil and Rint

# In[ ]:


import numpy
A = numpy.array(input().split(), float)
numpy.set_printoptions(sign=' ')  #risolve il solito bug degli spazi
print (numpy.floor(A))
print (numpy.ceil(A))
print (numpy.rint(A))


# # Sum and Prod

# In[ ]:


import numpy
n,m = map(int, input().split())
arr = numpy.array([input().split() for i in range(n)], int)
s = numpy.sum(arr, axis=0)
print(numpy.prod(s, axis=None))


# # Min and Max

# In[ ]:


import numpy
n, m = map(int, input().split())
arr = numpy.array([input().split() for i in range(n)], int)
m = numpy.min(arr, axis=1)
print(numpy.max(m))


# # Mean, Var and Std

# In[ ]:


import numpy
numpy.set_printoptions(legacy='1.13')  #versione numpy necessaria
n,m = map(int, input().split())
arr = numpy.array([input().split() for i in range(n)], int)
print(numpy.mean(arr, axis=1))
print(numpy.var(arr, axis=0))
print(numpy.std(arr, axis=None))


# # Dot and Cross

# In[ ]:


import numpy
n = int(input())
A = numpy.array([input().split() for i in range(n)], int)
B = numpy.array([input().split() for j in range(n)], int)

print(numpy.dot(A,B))


# # Inner and outer

# In[ ]:


import numpy
A = numpy.array(input().split(), int)
B = numpy.array(input().split(), int)

print(numpy.inner(A,B))
print(numpy.outer(A,B))


# # Pylonomials

# In[ ]:


import numpy
n = list(map(float,input().split()))
m = input()
print(numpy.polyval(n,int(m)))


# # Linear Algebra

# In[ ]:


import numpy
n = int(input())
numpy.set_printoptions(legacy='1.13')  #solito bug
A = numpy.array([input().split() for i in range(n)], float)

print(numpy.linalg.det(A))


# #### challenges of 2nd problem

# # Birthday Candles Cake

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the birthdayCakeCandles function below.



# SUPERA 4 TEST SU 8: running time problem
def birthdayCakeCandles(ar):
    count=0
    ar.sort()

    for i in ar[::-1]:
        if i == max(ar):
            count+=1
        if i != max(ar):
            break
    return count



# SUPERA 8 TEST SU 8
def birthdayCakeCandles(ar):
        cand = ar[0]
        count = 1
        for i in ar[1:]:
            if i > cand:
                cand = i
                count = 1
            elif i == cand:
                count+= 1
        return count




if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()


# # Kangaroos
# 

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys
# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    if ((x1>x2 and v1>v2) or (x1<x2 and v1<v2) or (v1-v2)==0):
        return "NO"
    if ((x1 - x2) % (v2 - v1)) == 0:
        return "YES"
    else:
        return "NO"
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


# # Viral Advertising

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
    share = 5
    liketot = 0

    for i in range(n):
        liketot = liketot + (share//2)
        share = (share//2)*3
    return liketot

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


# # Recursive Digit Sum

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):
    if len(n) == 1:
        return(n)
    somma =  sum([int(i) for i in str(n)])
    somma*= k
    return (superDigit(str(somma), 1))  

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

