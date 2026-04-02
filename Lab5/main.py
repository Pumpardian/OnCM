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

def vectorInput(n, type=float, restriction=lambda r: False):
    vec = []
    while True:
        user_input = input(f"Enter row: ").strip().split()
        if len(user_input) != n:
            print(f"{n} elements must be present. Try again...\n")
        else:
            try:
                for num in user_input:
                    if restriction(type(num)):
                        print("restriction not met, try again")
                        continue
                    vec.append(type(num))
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
    col_idx = i - 1
    l_hat = Q[:, col_idx]
    row_i = _A[col_idx, :]
    for r in range(n):
        factor = l_hat[r]
        if r == col_idx:
            factor -= 1
        if factor != 0:
            result[r, :] += factor * row_i
    return result

def calculate_inverse_matrix(A, _A, x, i):
    n = A.shape[0]
    A_asterisk = A.copy()
    A_asterisk[:, i-1] = x
    l = _A @ x
    if l[i-1] == 0:
        if np.linalg.matrix_rank(A_asterisk) < n:
            return A_asterisk, None, A_asterisk
        else:
            return None, None, None
    l_wave = l.copy()
    l_wave[i-1] = -1
    l_hat = (-1 / l[i-1]) * l_wave
    Q = np.identity(n)
    Q[:, i-1] = l_hat
    _A_asterisk = multiply_Q_A_optimized(Q, _A, n, i)
    return _A_asterisk, Q, A_asterisk

def dual_simplex_method(c, A, b, B_initial):
    m, n = A.shape
    B = [index - 1 for index in B_initial]
    method_iteration = 1

    #1 (First iter)
    AB = A[:,B]
    AB_inv = []
    try:
        AB_inv = np.linalg.inv(AB)
    except np.linalg.LinAlgError:
        print("Basis matrix AB cannot be inverted")
        return None
    
    while True:
        print(f"#ITERATION {method_iteration}")
        #1 (Non-first iter)
        if method_iteration != 1:
            AB_inv, _, AB = calculate_inverse_matrix(AB, AB_inv, A[:,j_0], k_index + 1)

        ###
        print(f"AB {AB}")
        print(f"AB Inverted {AB_inv}")
        ###

        #2
        cB = c[B]

        #3
        y = AB_inv.T @ cB

        ###
        print("cB =", cB)
        print("y^T = cB^T * AB_inv =", y)
        ###

        #4
        kappa_B = AB_inv @ b
        kappa = np.zeros(n)
        for i, bi in enumerate(B):
            kappa[bi] = kappa_B[i]

        ###
        print("κappa_B = AB_inv * b =", kappa_B)
        print("Pseudo-plan:\n", kappa)
        ###

        #5
        if np.all(kappa >= 0):
            ###
            print("Kappas are positive numbers, optimal plan has been found")
            ###

            return kappa
        else:
            #6
            negative_kappas = [index for index in range(m) if kappa_B[index] < 0]
            if not negative_kappas:
                ###
                print("Error: negative component wasn't found")
                ###

                return None
            
            k_index = negative_kappas[0]
            j_k = B[k_index]

            #7
            delta_y = AB_inv[k_index,:]

            ###
            print("delta_y =", delta_y)
            ###

            j_indicies = [j for j in range(n) if j not in B]
            nyu = {}
            for j in j_indicies:
                nyu[j] = delta_y @ A[:,j]

                ###
                print(f"nyu[{j + 1}] = delta_y^T * A[:, {j + 1}] = {nyu[j]}")
                ###

            #8
            if all(nyu[j] >= 0 for j in nyu):
                ###
                print("Task doesnt have a valid plan")
                ###

                return None
            
            #9
            sigma = {}
            for j in j_indicies:
                if nyu[j] < 0:
                    sigma[j] = (c[j] - (A[:,j] @ y)) / nyu[j]

                    ###
                    print(f"sigma[{j + 1}] = (c[{j + 1}] - A[:, {j + 1}]^T * y) / nyu[{j + 1}] = {sigma[j]}")
                    ###

            #10
            j_0, sigma_0 = min(sigma.items(), key=lambda item: item[1])

            ###
            print(f"j_0 with minimal sigma: sigma_0 = {sigma_0} when j_0 = {j_0 + 1}")
            ###

            #11
            B[k_index] = j_0

            ###
            print(f"Modifying basis: changing index {j_k + 1} to {j_0 + 1}.")
            print(f"New basis B:\n{[i + 1 for i in B]}")
            ###

            method_iteration += 1


if __name__ == "__main__":
    print("Enter matrix height (m > 0, number of restrictions): ")
    m = numberInput(int, lambda v: v > 0)
    print("Enter vectors size (n > 0, number of variables): ")
    n = numberInput(int, lambda v: v > 0)

    print("Enter target functional coefficients vector-column c: ")
    c = vectorInput(n)
    c = np.array([float(value) for value in c])

    print("Enter matrix A (constraint coefficients):")
    A = matrixInput(n, m)
    A = np.array(A, dtype=float)

    print("Enter right-part coefficients vector b: ")
    b = vectorInput(m)
    b = np.array([float(value) for value in b])

    print("Enter basis plan B")
    B = vectorInput(m, int, lambda v: v < 1 or v > n)
    B = [int(index) for index in B]

    print(f"Vector c:\n{c}\n")
    print(f"Matrix A:\n{A}\n")
    print(f"Vector b:\n{b}\n")
    print(f"Plan B:\n{B}\n")

    results = dual_simplex_method(c, A, b , B)
    if results is not None:
        print(f"Solution plan:\n{results}\n")
    else:
        print("No solution")