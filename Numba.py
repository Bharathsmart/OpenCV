#!/usr/bin/env python
# coding: utf-8

# In[16]:


import time
import numpy as np 
from numba import jit, float32

def sse(a):            # Without Numba
    result = 0 
    size = len(a)
    for i in range(size):
        result += a[i] * a[i]
    return result

#Dectator 
@jit(nopython = True)   # With Numba
def sse_numba(a):
    result = 0 
    size = len(a)
    for i in range(size):
        result += a[i] * a[i]
    return result
noel = int(1e5)
a = np.random.rand(noel)

result1 = sse_numba(a)  # With Numba
result2 = sse(a)     # Without Numba

print(result1)
print(result2)

num = 50    # Number of loops 

t1 = time.time()
for i in range(num):
    result2 = sse(a)
    
t2 = time.time()
for i in range(num):
    result1 = sse_numba(a) 
    
t3 = time.time()

print('Time without Numba :', t2-t1)
print('Time with Numba :', t3-t2)
print('Numba speeds up Python code by __ times :', (t2-t1)/(t3-t2))
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




