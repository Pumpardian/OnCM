import numpy as np

def numberInput(type=float, restriction=lambda r: True):
    while True:
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

def wolfe_method(Q, c, A, b):
    n = Q.shape[0]
    m = A.shape[0]
    
    # 1. Forming A_wave and b_wave
    # [x (n variables), v (n variables), u^+ (m variables), u^- (m variables)], in total: 2n + 2m variables
    O_mn = np.zeros((m, n))
    O_mm = np.zeros((m, m))
    E_n = np.eye(n)
    
    top_block = np.hstack([A, O_mn, O_mm, O_mm])
    bottom_block = np.hstack([Q, -E_n, A.T, -A.T])
    
    A_wave = np.vstack([top_block, bottom_block])
    b_wave = np.concatenate([b, -c])
    
    # 2. Preparing for the Phase 1 of Simplex method
    num_rows = m + n
    num_vars = 2 * n + 2 * m
    num_art = num_rows
    
    A_art = np.eye(num_art)
    
    for i in range(num_rows):
        if b_wave[i] < 0:
            A_wave[i, :] = -A_wave[i, :]
            A_art[i, :] = -A_art[i, :]
            b_wave[i] = -b_wave[i]
            
    # Evaluation row (delta) [delta_A_wave, delta_art]
    delta_A_wave = np.sum(A_wave, axis=0)
    delta_art = np.zeros(num_art)
    
    # Basis indexes array (starts with artificial variables)
    basis = list(range(num_vars, num_vars + num_art))
    
    # 3. Main loop of the modified Simplex method
    eps = 1e-9
    max_iters = 1000
    
    for iteration in range(max_iters):
        ###
        print(f"\nIteration #{iteration + 1}\n")
        print("A_wave (Constraints):\n", A_wave)
        print("b_wave (Right-Hand Side):", b_wave)
        print("delta_A_wave (Evaluations):", delta_A_wave)
        print("Current Basis Indexes:", basis)
        ###

        # Step 1: Choosing the entering variable (column)
        entering_col = -1
        best_delta = eps
        
        for j in range(num_vars):
            if delta_A_wave[j] > best_delta:
                # x_i and v_i cannot be in the basis at the same time (x_i * v_i = 0)
                if j < n:
                    if (n + j) in basis:
                        continue
                elif j >= n and j < 2 * n:
                    if (j - n) in basis:
                        continue
                
                entering_col = j
                best_delta = delta_A_wave[j]
                
        if entering_col == -1:
            ###
            print("Optimal solution reached (No negative deltas left).")
            ###
            break

        ###
        print(f"Entering variable index (Column): {entering_col} (Delta = {best_delta:.4f})")    
        ###

        # Step 2: Choosing the leaving variable (row)
        thetas = []
        for i in range(num_rows):
            if A_wave[i, entering_col] > eps:
                thetas.append(b_wave[i] / A_wave[i, entering_col])
            else:
                thetas.append(float('inf'))
                
        theta_0 = min(thetas)
        if theta_0 == float('inf'):
            ###
            print("Error: The task is unbounded.")
            ###
            return None
            
        leaving_row = thetas.index(theta_0)
        leaving_var = basis[leaving_row]
        
        ###
        print(f"Leaving variable index: {leaving_var} (Row: {leaving_row})")
        ###

        # Step 3: Updating data (Jordan-Gauss)
        pivot = A_wave[leaving_row, entering_col]
        
        A_wave[leaving_row, :] /= pivot
        A_art[leaving_row, :] /= pivot
        b_wave[leaving_row] /= pivot
        
        for i in range(num_rows):
            if i != leaving_row:
                factor = A_wave[i, entering_col]
                A_wave[i, :] -= factor * A_wave[leaving_row, :]
                A_art[i, :] -= factor * A_art[leaving_row, :]
                b_wave[i] -= factor * b_wave[leaving_row]
                
        factor_delta = delta_A_wave[entering_col]
        delta_A_wave -= factor_delta * A_wave[leaving_row, :]
        delta_art -= factor_delta * A_art[leaving_row, :]
                
        basis[leaving_row] = entering_col

    # 4. Optimal plan for x
    x_opt = np.zeros(n)
    for i, var_idx in enumerate(basis):
        if var_idx < n: # Extract only variables related to x
            x_opt[var_idx] = b_wave[i]
            
    return x_opt


if __name__ == "__main__":
    print("Enter matrix height (m > 0, number of restrictions): ")
    m = numberInput(int, lambda v: v > 0)
    print("Enter vectors size (n > 0, number of variables): ")
    n = numberInput(int, lambda v: v > 0)

    print("Enter target functional coefficients vector c: ")
    c = vectorInput(n)
    c = np.array([float(value) for value in c])

    print(f"Enter matrix Q (size: {n} x {n}):")
    Q = matrixInput(n, n)
    Q = np.array(Q, dtype=float)

    print("Enter matrix A (constraint coefficients):")
    A = matrixInput(n, m)
    A = np.array(A, dtype=float)

    print("Enter vector b:")
    b = vectorInput(m, int, lambda v: v < 1 or v > n)
    b = [int(index) for index in b]


    print(f"Vector c:\n{c}\n")
    print(f"Matrix Q:\n{Q}\n")
    print(f"Matrix A:\n{A}\n")
    print(f"Basis indexes b:\n{b}\n")

    x_optimal = wolfe_method(Q, c, A, b)
    if x_optimal is None:
        print("\nNo optimal plan was found")
    else:
        print(f"\nOptimal plan: {x_optimal}")