from copy import deepcopy
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
        matrix.append([float(num) for num in row])
    return np.array(matrix, dtype=float)

def place_marks_on_B(B_marks, basis):
    current_sign = not B_marks[basis]
    for (i, j) in B_marks.keys():
        if basis[0] == i or basis[1] == j:
            if B_marks[(i, j)] is None:
                B_marks[(i, j)] = current_sign
                place_marks_on_B(B_marks, (i, j))


def transport_task(supply, demand, cost):
    total_supply = np.sum(supply)
    total_demand = np.sum(demand)

    if total_supply > total_demand:
        demand = np.append(demand, total_supply - total_demand)
        cost = np.hstack((cost, np.zeros((len(supply), 1))))
        print("Balancing: added column (demand)")
    elif total_supply < total_demand:
        supply = np.append(supply, total_demand - total_supply)
        cost = np.vstack((cost, np.zeros((1, len(demand)))))
        print("Balancing: added row (supply)")

    m = len(supply)
    n = len(demand)

    # Building starting solution using "north-west corner" method 
    x = np.zeros((m, n))
    B = []
    i = 0
    j = 0
    supply_left = supply.copy()
    demand_left = demand.copy()

    while i < m and j < n:
        allocation = min(supply_left[i], demand_left[j])
        x[i, j] = allocation
        B.append((i, j))
        supply_left[i] -= allocation
        demand_left[j] -= allocation

        if i == m - 1 and j == n - 1:
            break

        if np.isclose(supply_left[i], 0) and i < m - 1:
            i += 1
        elif np.isclose(demand_left[j], 0) and j < n - 1:
            j += 1
        else:
            if i < m - 1:
                i += 1
            if j < n - 1:
                j += 1

    ###
    print("\nStarting solution:")
    print(f"Matrix x: {x}")
    print("Starting basis:", B)
    ###

    iteration = 0
    while True:
        iteration += 1
        print(f"Iteration #{iteration}")

        # Generating equasion system: u[i] + v[j] = cost[i, j]
        A_eq = []
        b_eq = []
        for (i_cell, j_cell) in B:
            row = [0] * (m + n)
            row[i_cell] = 1
            row[m + j_cell] = 1
            A_eq.append(row)
            b_eq.append(cost[i_cell, j_cell])

        # Additional equasion: u[0] = 0
        eq_fix = [0] * (m + n)
        eq_fix[0] = 1
        A_eq.append(eq_fix)
        b_eq.append(0)

        A_eq = np.array(A_eq, dtype=float)
        b_eq = np.array(b_eq, dtype=float)

        potentials = np.linalg.solve(A_eq, b_eq)
        u = potentials[:m]
        v = potentials[m:]

        ###
        print("\nu:", u)
        print("\nv:", v)
        ###

        # Min-eval search (for improving the plan):
        min_eval = None
        for i_cell in range(m):
            for j_cell in range(n):
                if (i_cell, j_cell) not in B and (cost[i_cell, j_cell] - u[i_cell] - v[j_cell]) < 0:
                    min_eval = (i_cell, j_cell)
                    break
            if min_eval is not None:
                break

        if min_eval is None:
            print(f"\nOptimal plan found: {x}")
            return

        print("\nMin_eval for addition into basis:", min_eval)

        # Addidng min_eval into basis and generating rearrangement cycle
        B.append(min_eval)
        B.sort()
        cycle = deepcopy(B)

        # Selecting and removing cells, that have more than 1 basis position in a row/column
        for i_cell in range(m):
            count = sum(1 for (p, q) in cycle if p == i_cell)
            if count <= 1:
                cycle = [cell for cell in cycle if cell[0] != i_cell]
        for j_cell in range(n):
            count = sum(1 for (p, q) in cycle if q == j_cell)
            if count <= 1:
                cycle = [cell for cell in cycle if cell[1] != j_cell]

        # Assigning signs to cycle cells: min_eval receiving "+" sign
        cycle_marks = {cell: None for cell in cycle}
        cycle_marks[min_eval] = True
        place_marks_on_B(cycle_marks, min_eval)

        # Calculating theta — minimal value from cells with negative sign
        theta = np.inf
        for cell, sign in cycle_marks.items():
            if sign is False:
                i_cell, j_cell = cell
                if x[i_cell, j_cell] < theta:
                    theta = x[i_cell, j_cell]
        print("\nTheta =", theta)

        # Plan correction: increasing values for cells with "+" and decreasing for cells with "-"
        for cell, sign in cycle_marks.items():
            i_cell, j_cell = cell
            if sign:
                x[i_cell, j_cell] += theta
            else:
                x[i_cell, j_cell] -= theta

        # If basis cell turned into 0, remove that cell from basis (excluding min_eval)
        for cell in B:
            i_cell, j_cell = cell
            if np.isclose(x[i_cell, j_cell], 0) and cell != min_eval:
                print("Removing cell:", cell)
                B.remove(cell)
                break

        print(f"\nUpdated plan x: {x}")
        print("Current basis:", B)


if __name__ == "__main__":
    print("Enter supply count: ")
    m = numberInput(int, lambda v: v > 0)
    print("Enter demand count: ")
    n = numberInput(int, lambda v: v > 0)

    print("Enter supply vector: ")
    s = vectorInput(m)
    supply = np.array([float(value) for value in s])

    print("Enter demand vector: ")
    d = vectorInput(n)
    demand = np.array([float(value) for value in d])

    print("Enter cost matrix: ")
    cost = matrixInput(n, m)

    results = transport_task(supply, demand, cost)