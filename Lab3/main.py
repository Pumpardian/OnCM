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

    return np.array(vec, float)

def matrixInput(n, m):
    matrix = []
    for i in range(1, m+1):
        print(f"Enter matrix's #{i} row")
        row = vectorInput(n)
        matrix.append(row)
    return np.array(matrix, float)

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
        return None, None, None
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


def main_simplex_method(c, A, x, B):
    m, n = A.shape
    method_iteration = 1

    AB = A[:,B]
    AB_inv = []
    try:
        AB_inv = np.linalg.inv(AB)
    except np.linalg.LinAlgError:
        print("Basis matrix AB cannot be inverted")
        return None
    
    while True:
        if method_iteration != 1:
            AB_inv, _, AB = calculate_inverse_matrix(AB, AB_inv, AB[:,j0], k)
        
        cB = c[B]
        
        u = cB @ AB_inv

        delta = u @ A - c

        print_iteration(method_iteration, AB, AB_inv, x, B, cB, u, delta)

        if np.all(delta >= 0):
            return x
        
        j0_list = np.where(delta < 0)[0]
        if len(j0_list) == 0:
            return x
        
        j0 = j0_list[0]
        Aj0 = A[:,j0]
        z = AB_inv @ Aj0

        theta = [x[B[i]] / z[i]     if z[i] > 0 else
                 np.inf
                 for i in range(m)]
        
        theta0 = np.min(theta)

        if theta0 is np.inf:
            print("Function is'nt limited")
            return None
        
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


if __name__ == "__main__":
    print("Enter matrix height (m > 0, number of restrictions): ")
    m = numberInput(int, lambda v: v > 0)
    print("Enter vectors size (n > 0, number of variables): ")
    n = numberInput(int, lambda v: v > 0)

    print("Enter target functional coefficients vector-column c: ")
    c = vectorInput(n)

    print("Enter matrix A (constraint coefficients):")
    A = matrixInput(n, m)

    print("Enter basis plan coefficients vector x (B set will be based on this vector): ")
    x = vectorInput(n)
    B = []
    for i in range(len(x)):
        if x[i] != 0:
            B.append(i)


    print(f"Vector c:\n{c}\n")
    print(f"Matrix A:\n{A}\n")
    print(f"Vector x:\n{x}\n")
    print(f"Set B:\n{B}\n")

    x_result = main_simplex_method(c, A, x , B)
    print(f"Resulting Vector x:\n{x_result}\n")