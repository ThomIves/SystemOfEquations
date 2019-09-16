# Linear Regression - Library Free, i.e. no numpy or scipy 


def check_squareness(A):
    """
    Makes sure that a matrix is square
        :param A: The matrix to be checked.
    """
    if len(A) != len(A[0]):
        raise ArithmeticError("Matrix must be square for current methods.")

def determinant(A, total=0):
    indices = list(range(len(A)))
    
    if len(A) == 2 and len(A[0]) == 2:
        val = A[0][0] * A[1][1] - A[1][0] * A[0][1]
        return val

    for fc in indices:
        As = copy_matrix(A)
        As = As[1:]
        height = len(As)
        builder = 0

        for i in range(height):
            As[i] = As[i][0:fc] + As[i][fc+1:]

        sign = (-1) ** (fc % 2)
        sub_det = determinant(As)
        total += A[0][fc] * sign * sub_det

    return total

def check_non_singular(A):
    det = determinant(A)
    if det != 0:
        return det
    else:
        raise ArithmeticError("Singular Matrix!")
        
def zeros_matrix(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have

        :returns: list of lists that form the matrix.
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M

def identity_matrix(n):
    """
    Creates and returns an identity matrix.
        :param n: the square size of the matrix

        :returns: a square identity matrix
    """
    I = zeros_matrix(n, n)
    for i in range(n):
        I[i][i] = 1.0

    return I

def copy_matrix(M):
    """
    Creates and returns a copy of a matrix.
        :param M: The matrix to be copied

        :return: The copy of the given matrix
    """
    rows = len(M)
    cols = len(M[0])

    MC = zeros_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]

    return MC

def print_matrix(M):
    """
    docstring here
        :param M: The matrix to be printed
    """
    for row in M:
        print([round(x,3)+0 for x in row])

def transpose(M):
    """
    Creates and returns a transpose of a matrix.
        :param M: The matrix to be transposed

        :return: the transpose of the given matrix
    """
    rows = len(M)
    cols = len(M[0])

    MT = zeros_matrix(cols, rows)

    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

    return MT

def matrix_multiply(A,B):
    """
    Returns the product of the matrix A * B
        :param A: The first matrix - ORDER MATTERS!
        :param B: The second matrix

        :return: The product of the two matrices
    """
    rowsA = len(A)
    colsA = len(A[0])

    rowsB = len(B)
    colsB = len(B[0])

    if colsA != rowsB:
        raise ArithmeticError('Number of A columns must equal number of B rows.')

    C = zeros_matrix(rowsA, colsB)

    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total

    return C

def check_matrix_equality(A,B, tol=None):
    """
    Checks the equality of two matrices.
        :param A: The first matrix
        :param B: The second matrix
        :param tol: The decimal place tolerance of the check

        :return: The boolean result of the equality check
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False

    for i in range(len(A)):
        for j in range(len(A[0])):
            if tol == None:
                if A[i][j] != B[i][j]:
                    return False
            else:
                if round(A[i][j],tol) != round(B[i][j],tol):
                    return False

    return True

def solve_equations(A, B, tol=None):
    """
    Returns the solution of a system of equations in matrix format.
        :param A: The system matrix

        :return: The solution X where AX = B
    """
    # Section 1: Make sure A can be inverted.
    check_squareness(A)
    check_non_singular(A)

    # Section 2: Make copies of A & I, AM & IM, to use for row operations
    n = len(A)
    # because of the need to reorder rows to avoid zero diagonal entries, we could just fill AM and BM with zero matrix here.
    AM = copy_matrix(A)
    I = identity_matrix(n)
    BM = copy_matrix(B)

    # Section 3: Perform row operations
    indices = list(range(n)) # to allow flexible row referencing ***
    
    # make sure diagonal has non-zero entries
    # make a list of the legal positions for each row (such that the element in the diagonal is non-zero)
    # do a depth first search to find a valid row ordering with non-zero diagonal entries,
    # which is guaranteed by non-singularity
    legal_row_positions = [filter(lambda y: tol == None and AM[x][y] != 0 or round(AM[x][y], tol) != 0, range(n))
                           for x in range(n)]
    row_choices = [-1] # this is needed to initialize the search properly.
    rows_taken = set()
    # depth first search
    while True:
        i = len(row_choices) - 1
        if row_choices[i] > -1:
            rows_taken.remove(legal_row_positions[i][row_choices[i]])
        row_choices[i] += 1
        while row_choices[i] < len(legal_row_positions[i]):
            if legal_row_positions[i][row_choices[i]] not in rows_taken:
                break
            row_choices[i] += 1
        if row_choices[i] < len(legal_row_positions[i]):
            rows_taken.add(legal_row_positions[i][row_choices[i]])
            if len(row_choices) == n: # finished
                break
            row_choices.append(-1) # descend
        else:
            row_choices.pop() # ascend
            if len(row_choices) == 0: # failure
                raise ValueError("Unexpectedly could not find a way to order rows with nonzero diagonal entries.")
    row_map = [legal_row_positions[i][row_choices[i]] for i in range(n)]
    # use the original (uncopied) arrays to rebuild the copies in the new order.
    for i in range(n):
        x = row_map[i]
        for j in range(n):
            AM[i][j] = A[x][j]
        BM[i] = B[x]
               
    for fd in range(n): # fd stands for focus diagonal
        fdScaler = 1.0 / AM[fd][fd]
        # FIRST: scale fd row with fd inverse. 
        for j in range(n): # Use j to indicate column looping.
            AM[fd][j] *= fdScaler
        BM[fd][0] *= fdScaler
        # SECOND: operate on all rows except fd row as follows:
        for i in indices[0:fd] + indices[fd+1:]: # *** skip row with fd in it.
            crScaler = AM[i][fd] # cr stands for "current row".
            for j in range(n): # cr - crScaler * fdRow, but one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
            BM[i][0] = BM[i][0] - crScaler * BM[fd][0]

    # Section 4: Make sure that BM is the solution for X
    if check_matrix_equality(B,matrix_multiply(A,BM),tol):
        return BM
    else:
        raise ArithmeticError("Solution for X out of tolerance.")
