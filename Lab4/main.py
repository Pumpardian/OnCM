import numpy as np

def numberInput(type=float, restriction=lambda r: True):
    while (True):
        try:
            element = type(input())
            if not restriction(element):
                print("restriction not met, try again")
                continue
            return element
        except ValueError:
            print(f"Invalid input. Enter valid {type} number")

def vectorInput(n):
    vec = []
    while True:
        user_input = input(f"Enter row: ").strip().split()
        if len(user_input) != n:
            print(f"{n} elements must be present. Try again...\n")
        else:
            try:
                for num in user_input:
                    vec.append(float(num))
                break
            except ValueError:
                print("Error: enter valid number values")
                vec = []
    return vec

def matrixInput(n, m):
    matrix = []
    for i in range(1, m+1):
        print(f"Enter matrix's #{i} row")
        row = vectorInput(n)
        matrix.append(row)
    return matrix

def multiply_Q_A_optimized(Q, _A, n, i):
    result = _A.copy()
    col_idx = i-1
    for row in range(n):
        result[row, col_idx] = 0
        for k in range(n):
            result[row, col_idx] += Q[row, k] * _A[k, col_idx]
    return result

def calculate_inverse_matrix(A, _A, x, i):
    n = A.shape[0]
    A_asterisk = A.copy()
    A_asterisk[:, i-1] = x
    l = _A @ x
    if l[i-1] == 0:
        if np.linalg.matrix_rank(A_asterisk) < n:
            return A_asterisk, None, A_asterisk
    l_wave = l.copy()
    l_wave[i-1] = -1
    l_hat = (-1 / l[i-1]) * l_wave
    Q = np.identity(n)
    Q[:, i-1] = l_hat
    _A_asterisk = multiply_Q_A_optimized(Q, _A, n, i)
    
    return _A_asterisk, Q, A_asterisk

def print_iteration(iter, AB, AB_inv, x, B, cB, u, delta,
                    j0=None, Aj0=None, z=None, theta=None, theta0=None, k=None, j_asterisk=None):
    print(f"ITERATION #{iter}")
    print(f"x:\n{x}\n")
    print(f"B:\n{B}\n")
    print(f"AB:\n{AB}\n")
    print(f"AB_inv:\n{AB_inv}\n")
    print(f"cB:\n{cB}\n")
    print(f"u:\n{u}\n")
    print(f"delta:\n{delta}\n")
    if (j0 is not None):
        print(f"j0:\n{j0}\n")
        print(f"Aj0:\n{Aj0}\n")
        print(f"z:\n{z}\n")
        print(f"theta:\n{theta}\n")
        print(f"theta0:\n{theta0}\n")
        print(f"k:\n{k}\n")
        print(f"j_asterisk:\n{j_asterisk}\n")
    print(f"{'-'*16}")


def main_simplex_method(c, A, x_init, B_init):
    m, n = A.shape
    method_iteration = 1

    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    x = np.array(x_init, dtype=float)
    B = [b_index - 1 for b_index in B_init]

    AB = A[:,B]
    AB_inv = []
    try:
        AB_inv = np.linalg.inv(AB)
    except np.linalg.LinAlgError:
        print("Basis matrix AB cannot be inverted")
        return None, None, None
    
    while True:
        if method_iteration != 1:
            AB_inv, _, AB = calculate_inverse_matrix(AB, AB_inv, AB[:,j0], k)
        
        cB = c[B]

        u = cB @ AB_inv

        delta = u @ A - c

        print_iteration(method_iteration, AB, AB_inv, x, B, cB, u, delta)

        if np.all(delta >= 0):
            return x, B, AB_inv
        
        j0_list = np.where(delta < 0)[0]
        if len(j0_list) == 0:
            return x, B, AB_inv
        
        j0 = j0_list[0]
        Aj0 = A[:,j0]
        z = AB_inv @ Aj0

        theta = [x[B[i]] / z[i]     if z[i] > 0 else
                 np.inf
                 for i in range(m)]
        
        theta0 = np.min(theta)

        if theta0 is np.inf:
            print("Function is'nt limited")
            return None, None, None
        
        k = np.where(np.array(theta) == theta0)[0][0]
        j_asterisk = B[k]

        B_new = B.copy()
        B_new[k] = j0

        x_new = x.copy()
        x_new[j0] = theta0
        for i in range(m):
            if i is not k:
                x_new[B[i]] = x_new[B[i]] - theta0 * z[i]
        x_new[j_asterisk] = 0

        print_iteration(method_iteration, AB, AB_inv, x_new, B_new, cB, u, delta,
                        j0, Aj0, z, theta, theta0, k, j_asterisk)

        x = x_new
        B = B_new
        method_iteration += 1

def initial_simplex_method(c, A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape

    #1
    for i in range(m):
        if b[i] < 0:
            A[i,:] = -A[i,:]
            b[i] = -b[i]
    
    #2
    c_wave = np.hstack((np.zeros(n), -np.ones(m)))
    A_wave = np.hstack((A, np.eye(m)))
    
    #3
    x_wave_initial = np.concatenate((np.zeros(n), b))
    B_wave_initial = [i for i in range(n + 1, n + m + 1)]

    #4
    x_wave, B_wave, AB_inv = main_simplex_method(c_wave, A_wave, x_wave_initial, B_wave_initial)

    #5
    if x_wave is None:
        print("Auxilary task cannot be solved")
        return None, None, None, None
    
    artificial_values = x_wave[n:]
    if not np.all(np.abs(artificial_values) < 1e-8):
        print(
            "\nError: artifitial variables are not 0, there is no solution"
        )
        return None, None, None, None

    #6
    x = x_wave[:n]

    #7
    while any(b_index >= n for b_index in B_wave):
        artificial_in_B = [b_index for b_index in B_wave if b_index >= n]
        
        #8
        j_k = max(artificial_in_B)
        k = B_wave.index(j_k)
        i_row = j_k - n

        #9
        replaced = False
        for j in range(n):
            if j in B_wave:
                continue
            l_j = AB_inv @ A_wave[:, j]
            
            #10
            if not np.isclose(l_j[k], 0.0):
                print(f"  Replacing {j_k + 1} with {j + 1} (l(j))[{k + 1}] = {l_j[k]:.4f}")
                B_wave[k] = j
                AB = A_wave[:, B_wave]
                try:
                    AB_inv = np.linalg.inv(AB)
                except np.linalg.LinAlgError:
                    print("Error: basis after replacement cannot be inverted")
                    return None, None, None, None, None
                replaced = True
                break
        
        #11
        if not replaced:
            print(
                f"  All j in B is applicable to (l(j))[{k + 1}] = 0. Deleting constraint {i_row + 1}."
            )
            A = np.delete(A, i_row, axis=0)
            b = np.delete(b, i_row, axis=0)
            A_wave = np.delete(A_wave, i_row, axis=0)
            m -= 1
            del B_wave[k]
            if len(B_wave) > 0:
                AB = A_wave[:, B_wave]
                try:
                    AB_inv = np.linalg.inv(AB)
                except np.linalg.LinAlgError:
                    print("Error: basis cannot be inverted after deletion")
                    return None, None, None, None, None

    B_final = [int(b_idx) + 1 for b_idx in B_wave]
    return x, B_final, A, b

if __name__ == "__main__":
    print("Enter matrix height (m > 0, number of restrictions): ")
    m = numberInput(int, lambda v: v > 0)
    print("Enter vectors size (n > 0, number of variables): ")
    n = numberInput(int, lambda v: v > 0)

    print("Enter target functional coefficients vector-column c: ")
    c = vectorInput(n)

    print("Enter matrix A (constraint coefficients):")
    A = matrixInput(n, m)

    print("Enter right-parts vector vector b:")
    b = vectorInput(m)

    print(f"Vector c:\n{c}\n")
    print(f"Matrix A:\n{A}\n")
    print(f"Vector b:\n{b}\n")

    x, B_final, A, b = initial_simplex_method(c, A, b)
    if x is None:
        print("\nNo plans available")
    else:
        print("\nInitial phase results:")
        print("x:")
        print(x)
        print("B:")
        print(B_final)
        print("A:")
        print(A)
        print("b:")
        print(b)