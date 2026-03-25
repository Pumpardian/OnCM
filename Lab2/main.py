import numpy as np

def numberInput(type=float):
    while (True):
        try:
            element = type(input())
            return element
        except ValueError:
            print(f"Invalid input. Enter valid {type} number")

def vectorInput(n, signs=False):
    vec = []
    while True:
        user_input = input(f"Enter row: ").strip().split()
        if len(user_input) != n:
            print(f"{n} elements must be present. Try again...\n")
        else:
            try:
                if signs:
                    for num in user_input:
                        val = float(num)
                        if val > 0:
                            vec.append(1)
                        elif val < 0:
                            vec.append(-1)
                        else:
                            vec.append(0)
                    break
                else:
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


def linear_to_normal(c, d, A, b, r, s):
    print("1. Checking min or max, if min then c *= -1")
    if not d:
        c = -c

    A_new = A.copy()
    b_new = b.copy()
    c_new = c.copy()
    r_new = r.copy()
    s_new = s.copy()

    print("2. For each r[i] = \"=\" splitting in two")
    new_rows = []
    new_b = []
    new_r = []
    
    for i in range(len(r_new)):
        if r_new[i] == 0:
            new_rows.append(A_new[i])
            new_b.append(b_new[i])
            new_r.append(-1)

            new_rows.append(A_new[i])
            new_b.append(b_new[i])
            new_r.append(1)
        else:
            new_rows.append(A_new[i])
            new_b.append(b_new[i])
            new_r.append(r_new[i])
    
    A_new = np.array(new_rows)
    b_new = np.array(new_b)
    r_new = np.array(new_r)

    print("3. For each r[i] = \">=\" inverting to \"<=\"")
    for i in range(len(r_new)):
        if r_new[i] == 1:
            A_new[i] *= -1
            b_new[i] *= -1
            r_new[i] = -1

    print("4. For each s[i] = \"<=\" inverting")
    for i in range(len(s_new)):
        if s_new[i] == -1:
            A_new[:, i] *= -1
            c_new[i] *= -1
            s_new[i] = 1

    print("5. For each s[i] = \"><\" splitting in two")
    final_cols = []
    final_c = []
    
    for i in range(len(s_new)):
        if s_new[i] == 1:
            final_cols.append(A_new[:, i:i+1])
            final_c.append(c_new[i])
        else:
            final_cols.append(A_new[:, i:i+1])
            final_c.append(c_new[i])
            final_cols.append(-A_new[:, i:i+1])
            final_c.append(-c_new[i])

    if final_cols:
        A_new = np.hstack(final_cols)
        c_new = np.array(final_c)

    return c_new, A_new, b_new


def linear_to_canonical(c, d, A, b, r, s):
    print("1. Checking min or max, if min then c *= -1")
    if not d:
        c = -c

    print(f"Vector c:\n{c}\n")

    A_new = A.copy()
    b_new = b.copy()
    c_new = c.copy()
    r_new = r.copy()
    s_new = s.copy()

    print("2-3. For each not \"=\" add columns and append to s")
    extra_cols = []
    extra_c = []
    extra_s = []

    for i in range(len(r_new)):
        if r_new[i] == -1:
            col = np.zeros((A.shape[0], 1))
            col[i, 0] = 1
            extra_cols.append(col)
            extra_c.append(0)
            extra_s.append(1)
            r_new[i] = 0
        elif r_new[i] == 1:
            col = np.zeros((A.shape[0], 1))
            col[i, 0] = -1
            extra_cols.append(col)
            extra_c.append(0)
            extra_s.append(1)
            r_new[i] = 0

    if extra_cols:
        A_new = np.hstack((A_new, np.hstack(extra_cols)))
        c_new = np.append(c_new, extra_c)
        s_new = np.append(s_new, extra_s)

    print("4. For each s[i] = \"<=\" inversing")
    for i in range(len(s_new)):
        if s_new[i] == -1:
            A_new[:, i] *= -1
            c_new[i] *= -1
            s_new[i] = 1

    print("5. For each s[i] = \"><\" splitting in two")
    final_cols = []
    final_c = []
    
    for i in range(len(s_new)):
        if s_new[i] == 1:
            final_cols.append(A_new[:, i:i+1])
            final_c.append(c_new[i])
        else:
            final_cols.append(A_new[:, i:i+1])
            final_c.append(c_new[i])
            final_cols.append(-A_new[:, i:i+1])
            final_c.append(-c_new[i])

    if final_cols:
        A_new = np.hstack(final_cols)
        c_new = np.array(final_c)

    return c_new, A_new, b_new


if __name__ == "__main__":
    print("Enter vectors size (number of variables): ")
    n = numberInput(int)
    print("Enter matrix height (number of functionals): ")
    m = numberInput(int)

    print("Enter vector-column c: ")
    c = vectorInput(n)

    d = bool(input("Choose whether it's minimization or maximization task (Leave blank for min, anything else for max): "))

    print("Enter matrix A (constraint coefficients):")
    A = matrixInput(n, m)

    print("Enter vector-column b: ")
    b = vectorInput(m)

    print("Enter vector-column r (signs of constraints: negative for <=, positive for >=, 0 for =): ")
    r = vectorInput(m, signs=True)

    print("Enter vector-column s (signs of variables: psoitive for >= 0, negative for <= 0, 0 for ><): ")
    s = vectorInput(n, signs=True)

    t = bool(input("Choose, to normal or to canonical form (Blank for normal): "))
    
    if not t:
        c, A, b = linear_to_normal(c, d, A, b, r, s)
    else:
        c, A, b = linear_to_canonical(c, d, A, b, r, s)

    print(f"Vector c:\n{c}\n")
    print(f"Matrix A:\n{A}\n")
    print(f"Vector b:\n{b}\n")