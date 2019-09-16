#!/usr/bin/env python
# coding: utf-8

# ![Matrix Inversion Logo](System_of_Equations_Logo.png)
# # Dirt Simple Solution of a System of Equations
# [SystemOfEquations on Github](https://github.com/ThomIves/SystemOfEquations)

# We are going to walk thru a brute force procedural method for solving a system of equations with pure Python. Why wouldn’t we just use numpy? Great question. This blog is about tools that add efficiency **_AND_** _clarity_. I love numpy, pandas, sklearn, and all the great tools that the python data science community brings to us, but I have learned that the better I understand the “principles” of a thing, the better I know how to apply it. Plus, *tomorrow's machine learning tools will be developed by those that understand the **principles** of the math and coding of today’s tools.* 
# 
# Also, once an efficient method of solving a system of equations is understood, you are ~ 80% of the way to having your own Least Squares Solver and a component to many other personal analysis modules to help you better understand how all our great machine learning tools are built. Would I recommend that you use what we are about to develop for a real project? All those python modules mentioned previously are lightening fast, so, usually, no. I would not recommend that you use your own such tools **UNLESS** you are working with smaller problems, **OR** you are investigating some new approach that requires slight changes to your personal tool suite. Thus, a statement above bears repeating: *tomorrows machine learning tools will be developed by those that understand the **principles** of the math and coding of today’s tools.* I want to be part of, or at least foster, those that will make the next gen tools. Plus, if you are a geek, knowing how to code the solution to a system of equations is fun!
# 
# The way that I was taught to solve a system of equations was to find the inverse of the A matrix (system matrix) first, and the way that I was taught to do that, *in the dark ages that is*, was pure torture! If you go inverting a matrix the way that you would program it, it is MUCH easier in my opinion, AND it directly relates to how to solve the system of equations directly. I would even think it’s easier doing the method we will use when doing it by hand than the ancient teaching of how to do it. In fact, it is so easy that we will start with a 5x5 matrix to make it “clearer”.
# 
# **DON’T PANIC.** The only really painful thing about it, is that, while it’s very simple, it’s a bit tedious and boring. However, compared to the ancient method, it’s simple. Or, as one of my favorite mentors would commonly say, “It’s simple. it’s just not easy.” We’ll use python, to reduce the tedium, without losing any view to the insights of the methods.
# 
# We’ll use python at first through a Jupyter notebook to clearly illustrate each step. Then, we’ll be VERY ready to adapt those steps to build our own module. I will seek to be very pep8’ish. Please deviate from my style as you wish to make what we are doing your own and more clear to you. You’ll be glad that you did.
# 
# The logo for the github repo that stores all this work, really says it all.
# 
# Following the main rule of algebra (whatever we do to one side of the equal sign, we will do to the other side of the equal sign, in order to “stay true” to the equal sign), we will perform row operations to **A** in order to methodically turn it into an identity matrix while applying those same steps to what is “initially” the B matrix on the right (i.e. the answer matrix). 
# 
# When what was __A__ becomes an identity matrix, what was B on the right will become the solution for **X**. 
# 
# If at some point, you have a big **“Ah HA!”** moment, try to work ahead on your own and compare to what we’ve done once you’ve finished or if you get stuck. 
# 
# The Jupyter notebook called **SystemOfEquations.ipynb** can be obtained from the [github repo](https://github.com/ThomIves/MatrixInverse) for this project. You don’t need to use Jupyter notebooks to follow along. I’ve also saved the cells as SystemOfEquations.py in the same repo. Let’s first define some helper functions.

# In[32]:


def print_matrix(Title, M):
    print(Title)
    for row in M:
        print([round(x,3)+0 for x in row])
        
def print_matrices(Action, Title1, M1, Title2, M2):
    print(Action)
    print(Title1, '\t'*int(len(M1)/2)+"\t"*len(M1), Title2)
    for i in range(len(M1)):
        row1 = ['{0:+7.3f}'.format(x) for x in M1[i]]
        row2 = ['{0:+7.3f}'.format(x) for x in M2[i]]
        print(row1,'\t', row2)
        
def zeros_matrix(rows, cols):
    A = []
    for i in range(rows):
        A.append([])
        for j in range(cols):
            A[-1].append(0.0)

    return A

def copy_matrix(M):
    rows = len(M)
    cols = len(M[0])

    MC = zeros_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]

    return MC

def matrix_multiply(A,B):
    rowsA = len(A)
    colsA = len(A[0])

    rowsB = len(B)
    colsB = len(B[0])

    if colsA != rowsB:
        print('Number of A columns must equal number of B rows.')
        sys.exit()

    C = zeros_matrix(rowsA, colsB)

    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total

    return C


# **NOTE:** The last print statement uses a trick to get rid of -0.0’s. Try it with and without the “+0” to see what I mean.
# 
# Let’s prepare some matrices to use.

# In[33]:


# A = [[5.,4.,3.,2.,1.],[4.,3.,2.,1.,5.],[3.,2.,9.,5.,4.],[2.,1.,5.,4.,3.],[1.,2.,3.,4.,5.]]
# B = [[48],[38],[64],[43],[42]]
A = [[5.,3.,1.],[3.,9.,4.],[1.,3.,5.]]
B = [[9.0],[16.0],[9.0]]
print_matrices('','A Matrix', A, 'B Matrix', B)


# Let's make some copies that we can morph while preserving our originals

# In[34]:


AM = copy_matrix(A)
BM = copy_matrix(B)
n = len(AM)

exString = """
Since the matrices won't be the original A and I as we start row operations, 
    the matrices will be called: AM for "A Morphing", and BM for "B Morphing" 
"""
print_matrices(exString, 'AM Matrix', AM, 'BM Matrix', BM)
print()


# The first basic step, consider the first element of the diagonal of **AM**, which is __5__. Let's divide all elements of the first row by it. 
# 
# From here forward, all operations applied to **AM**, also are applied to __BM__.

# In[43]:


# Run this cell then the next for fd = 0, 1, 2, 3, and 4 for a 5x5 matrix. 
#      Then check for identity matrix in last cell.

fd = 1 # fd stands for focus diagonal OR the current diagonal
fdScaler = 1. / AM[fd][fd]

for j in range(n): # using j to indicate cycling thru columns
    AM[fd][j] = fdScaler * AM[fd][j]
BM[fd][0] = fdScaler * BM[fd][0]
    
print()
print_matrices('', 'AM Matrix', AM, 'BM Matrix', BM)


# Now we do the following: 
# 1. look at the rows below the focus diagonal element that we just scaled; 
# 2. for each of those rows, use the elements above and/or below the current focus diagonal as a scaler;
# 3. replace each row with the result of [current row] - scaler*[row with focus diagonal];
# 4. This leaves zeros in the column shared with the focus diagonal element, which was previously scaled to 1.
# 
# If you’re as big a geek as me, you have chills now.

# In[44]:


n = len(A)
indices = list(range(n))

for i in indices[0:fd] + indices[fd+1:]: # *** skip row with fd in it.
    crScaler = AM[i][fd] # cr stands for "current row".
    for j in range(n): # cr - crScaler * fdRow, but one element at a time.
        AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
    BM[i][0] = BM[i][0] - crScaler * BM[fd][0]
    
print_matrices('', 'AM Matrix', AM, 'BM Matrix', BM)
print()


# After cycling through all the diagonal elements of our morphing **AM** matrix, and performing the same steps on **BM** that we've done to __AM__, the __BM__ matrix will become the solution of **X** or __Xs__.

# In[45]:


# We need fresh copies now ...
AM = copy_matrix(A)
BM = copy_matrix(B)
n = len(AM)

indices = list(range(n)) # to allow flexible row referencing ***
for fd in range(0,n): # fd stands for focus diagonal
    fdScaler = 1.0 / AM[fd][fd]
    # FIRST: scale fd row with fd inverse. 
    for j in range(n): # Use j to indicate column looping.
        AM[fd][j] *= fdScaler
    BM[fd][0] *= fdScaler
    
    # Section to print out current actions:
    string1 = '\nUsing the matrices above, Scale row-{} of AM and IM by '
    string2 = 'diagonal element {} of AM, which is 1/{:+.3f}.\n'
    stringsum = string1 + string2
    val1 = fd+1; val2 = fd+1
    Action = stringsum.format(val1,val2,round(1./fdScaler,3))
    print_matrices(Action, 'AM Matrix', AM, 'BM Matrix', BM)
    print()
    
    # SECOND: operate on all rows except fd row.
    for i in indices[:fd] + indices[fd+1:]: # *** skip row with fd in it.
        crScaler = AM[i][fd] # cr stands for "current row".
        for j in range(n): # cr - crScaler * fdRow, but one element at a time.
            AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
        BM[i][0] = BM[i][0] - crScaler * BM[fd][0]
        
        # Section to print out current actions:
        string1 = 'Using the matrices above, subtract {:+.3f} * row-{} of AM from row-{} of AM, and \n'
        string2 = '\tsubtract {:+.3f} * row-{} of BM from row-{} of BM\n'
        val1 = i+1; val2 = fd+1
        stringsum = string1 + string2
        Action = stringsum.format(crScaler, val2, val1, crScaler, val2, val1)
        print_matrices(Action, 'AM Matrix', AM, 'BM Matrix', BM)
        print()


# Success! 
# 
# **AM** has morphed into an Identity matrix. 
# 
# **BM** has become the solution for __X__. 
# 
# Yay! And yes, I am easily entertained. 

# Let’s apply some helper functions to accomplish a proof of the solution.

# In[46]:


# A = [[5,4,3,2,1],[4,3,2,1,5],[3,2,9,5,4],[2,1,5,4,3],[1,2,3,4,5]]
print_matrix('Proof of Solution', matrix_multiply(A,BM))


# Yes! When we perform
# __A * BM__
# we do get the expected solution for 
# **X**. 
# 
# I do love Jupyter notebooks, but I want to use this in scripts now too. See LinearAlgebraPurePython.py and a file that uses it - LinearAlgebraPractice.py.
