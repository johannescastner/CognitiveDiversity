def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or board[i] - i == col - row or board[i] + i == col + row:
                return False
        return True

    def solve(board, row):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solve(board, row + 1)
                board[row] = -1

    solutions = []
    solve([-1] * n, 0)
    return solutions

def print_solutions(solutions):
    for solution in solutions:
        for row in solution:
            print(' '.join(['Q' if col == row else '.' for col in range(len(solution))]))
        print("\n")

if __name__ == "__main__":
    n = 4  # Change this value to solve for different sizes
    solutions = solve_n_queens(n)
    print(f"Found {len(solutions)} solutions for {n} queens:")
    print_solutions(solutions)
