import LinearAlgebraPurePython as la 
import ShortImplementation as si


print('A matrix')
A = [[5,4,3,2,1],[4,3,2,1,5],[3,2,9,5,4],[2,1,5,4,3],[1,2,3,4,5]]
la.print_matrix(A)
print()

print('X matrix')
X = [[3],[4],[2],[5],[1]]
la.print_matrix(X)
print()

print('Calculate B from X')
B = la.matrix_multiply(A, X)
la.print_matrix(B)
print()

print('Solve for X')
XS = la.solve_equations(A,B,9)
la.print_matrix(XS)
print()

# print('Solve for X')
# XS = si.solve_equations(A,B)
# la.print_matrix(XS)
