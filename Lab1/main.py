import numpy as np


def matrix_input(n):
    X = []
    for row in range(1, n+1):
        while True:
            user_input = input(f"Row {row}: ").strip().split()
            if len(user_input) != n:
                print(f"{n} elements must be present. Try again...\n")
            else:
                try:
                    X.append([float(num) for num in user_input])
                    break
                except ValueError:
                    print("Error: enter valid number values")
    
    return np.array(X, float)


def multiply_Q_A_optimized(Q, _A, n, i):
    result = _A.copy()
    
    col_idx = i-1

    for row in range(n):
        result[row, col_idx] = 0
        for k in range(n):
            result[row, col_idx] += Q[row, k] * _A[k, col_idx]
    
    return result


def user_input():
    try:
        n = int(input("Enter matrix dimensions (n), matrix are NxN: "))
        if n <= 0:
            raise ValueError("Matrix dimension must be positive integer")
        
        print(f"Enter maxtrix A elements, {n} rows, use space as delimeter:\n")
        A = matrix_input(n)

        print(f"\nEnter maxtrix A^-1 elements, {n} rows, use space as delimeter:\n")
        _A = matrix_input(n)

        print(f"Enter vector x elements (size: {n}), use space as delimeter:\n")
        while True:
            x_input = input("x: ").strip().split()
            if len(x_input) != n:
                print(f"{n} elements must be present. Try again...")
            else:
                try:
                    x = np.array([float(number) for number in x_input], float)
                    break
                except ValueError:
                    print("Error: enter valid number values")

        while True:
            try:
                i = int(input("Enter index i value: "))
                if 1 <= i <= n:
                    break
                else:
                    print(f"Invalid index, must be in range [1; {n}], try again...\n")
            except ValueError:
                print(f"Error: i must be an positive integer [1; {n}], try again...\n")
        
        return A, _A, x, i
    except ValueError as e:
        print(f"Input error: {e}")
        exit(1)


def calculate_inverse_matrix(A, _A, x, i):
    n = A.shape[0]

    print("Calculating maxtrix A' by replacing i-th column with vector x")
    A_asterisk = A.copy()
    A_asterisk[:, i-1] = x
    print(f"Matrix A':\n{A_asterisk}\n")

    print("Then we'll calculate vector l = A^-1 * x")
    l = _A @ x
    print(f"Vector l:\n{l.reshape(-1,1)}\n")

    if l[i-1] == 0:
        print(f"l[{i}] = 0, matrix A' is uninversable")
        return None, None, None
    print(f"l[{i}] != 0, matrix A' is inversable")
    
    print("Calculating vector l~, by replacing i-th element with -1")
    l_wave = l.copy()
    l_wave[i-1] = -1
    print(f"Vector l~:\n{l_wave.reshape(-1,1)}\n")

    print("Calculating 'l = (-1 / (l[i])) * l~")
    l_hat = (-1 / l[i-1]) * l_wave
    print(f"Vector 'l:\n{l_hat.reshape(-1,1)}\n")

    print("Calculating matrix Q, i-th colums is 'l")
    Q = np.identity(n)
    Q[:, i-1] = l_hat
    print(f"Matrix Q:\n{Q}\n")

    print("Calculating (A')^-1 = Q * A^-1")
    _A_asterisk = multiply_Q_A_optimized(Q, _A, n, i)
    print(f"Matrix A'^-1:\n{_A_asterisk}\n")
    
    return _A_asterisk, Q, A_asterisk


def main():
    try:
        A, _A, x, i = user_input()

        print(f"Matrix A:\n{A}\n")
        print(f"Inverse matrix A^-1:\n{_A}\n")
        print(f"Vector x:\n{x.reshape(-1,1)}\n")
        print(f"Index i: {i}\n")

        _A_asterisk, Q, A_asterisk = calculate_inverse_matrix(A, _A, x, i)
        if _A_asterisk is not None:
            print(f"Resulting inversed matrix A'^-1 is:\n{_A_asterisk}")
            
    except Exception as e:
        print(f"Error occured: {e}")


if __name__ == "__main__":
    main()