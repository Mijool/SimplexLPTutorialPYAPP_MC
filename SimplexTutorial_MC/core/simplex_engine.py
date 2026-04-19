"""
Core mathematical engine for the Simplex Algorithm.
"""
import time as dt
from datetime import timedelta

import numpy as np

#underscore denotes a 'private' variable
class SimplexSolver:

    #def is a method, __init__ is a constructor, you must pass in self (equivalent of 'this' keyword)
    def __init__(self, obj_coefficients: list[float],
                 constraint_matrix: list[list[float]],
                 rhs_values: list[float],
                 maximize=True) -> None:

        # create arrays from our initial lists, these are our anchors that we only need to access but not manipulate during solve logic
        # (list being entered, data-type)

        # 1. Initialize original inputs - A and c get appended slack variables after wards
        self._maximize = maximize
        original_A = np.array(constraint_matrix, dtype=float)
        self._b = np.array(rhs_values, dtype=float)
        if maximize:
            original_c = np.array(obj_coefficients, dtype=float)
        else: #when minimizing, we maximize negative z
            original_c = np.array(obj_coefficients, dtype=float) * -1.0




        # 2. Get original dimensions
        self._num_constraints: int = original_A.shape[0] # of rows
        self._num_vars: int = original_A.shape[1] # of columns

        # 3. Append Slack Variables to create Standard Form
        # A becomes [A | I] and c becomes [c | 0]
        slack_A = np.eye(self._num_constraints, dtype=float)
        self._A = np.hstack((original_A, slack_A))  # Now A is a 3x5 matrix
        #horizontal stack(matrix 1 | matrix 2)

        slack_c = np.zeros(self._num_constraints, dtype=float)
        self._c = np.concatenate((original_c, slack_c))  # Now c is size 5
        #concatenate is better for vectors like c

        # 4. Setup dynamic variables (The Basis)
        self.B_inv = np.eye(self._num_constraints, dtype=float)

        # 5. Initial basis is the slack variables
        # This generates [2, 3, 4] for a 2-var, 3-constraint problem
        self.basic_indices: list[int] = list(range(self._num_vars, self._num_vars + self._num_constraints))

        self.cB: np.ndarray = np.zeros(self._num_constraints, dtype=float)
        self.status: str = "Setup Complete"

    def _find_y_star(self) -> np.ndarray:
        """finds y* (cB*B_inv), we pass our simplex problem in as the parameter (non-static method)"""
        y_star: np.ndarray = np.zeros(self._num_constraints, dtype=float)

        # @ operator does matrix multiplication
        y_star = self.cB @ self.B_inv
        return y_star
    def _find_entering_variable_index(self, y_star: np.ndarray) -> int:
        """finds entering variable using y*[A] - c, or returns -1 if optimal"""
        z_star: np.ndarray = np.zeros(self._num_constraints, dtype=float)
        z_star = (y_star @ self._A) - self._c
        entering_variable_index: int = int(np.argmin(z_star)) #argmin returns the index of the smallest value in the entered array

        if z_star[entering_variable_index] < 0: #if there is a negative number in the z row, its not optimal yet, and should enter the basis
            self.status = "Entering variable is x"+str(entering_variable_index)
            return entering_variable_index
        else: #otherwise, all variables are positive and the solution is optimal
            self.status = "Optimal solution found"
            return -1


    def _minimum_ratio_test(self, A_star: np.ndarray, b_star: np.ndarray) -> int:
        """calculate A* = S* @ A and b* = S* @ b to find row index of our leaving variable, return -1 if unbounded (all are negative or undefined)"""

        #we have to use a tiny number instead of zero to eliminate problems with floating-point errors
        # wrong: positive_leaving_indices = np.argwhere( ratio_arr > 0).flatten()
        valid_rows = np.argwhere(A_star > 1e-9).flatten()

        # These get calculated outside and entered in here
        # A_star: np.ndarray = (self.B_inv @ self._A), entering_col] # slices matrix to only get the column we need for our min. ratio test, ':' selects all rows, and we only want the entering variables column
        #
        # b_star: np.ndarray = self.B_inv @ self._b

        if valid_rows.size == 0:
            self.status = "No leaving variable, problem is considered unbounded"
            return -1

        #we get the ratios of only our positive rows
        ratio_arr = b_star[valid_rows] / A_star[valid_rows]

        #argmin gets the index, min gets the value of the smallest number after the ratio test
        min_valid_value = int(np.argmin(ratio_arr))

        #store the index where the leaving variable is located
        leaving_variable_index: int = int(valid_rows[min_valid_value])

        #leaving_variable_index: int = int( valid_rows[ratio_arr[ratio_arr > 0].argmin()] )  # we can easily divide matrices in this manner, we return the smallest value that isn't negative
        self.status = "Leaving variable index is "+str(leaving_variable_index)
        return leaving_variable_index

    def _build_new_B_inv (self, A_star: np.ndarray, leaving_row_index: int) -> np.ndarray:
        """build eta and E and returns E*B_inv for new B_inv (S*) """
        #previously was using the entering column and not the entire new matrix

        eta: np.ndarray = np.zeros(self._num_constraints, dtype=float)
        for i in range(len(A_star)):
            if i == leaving_row_index:
                eta[i] = 1.0/A_star[leaving_row_index]
            else:
                eta[i] = -1*(A_star[i])/A_star[leaving_row_index]

        E: np.ndarray = np.eye(self._num_constraints, dtype=float) #create an identity matrix in the shape of our constaints
        E[:, leaving_row_index] = eta #[select all rows, select the column that aligns with the row that is leaving] = eta vector

        S_star = E @ self.B_inv

        return S_star #returns S*

    def solve(self) -> tuple[np.ndarray, float, str]:
        """
        Executes the Revised Simplex loop until optimality or unboundedness is reached.
        Returns (final_solution_values, max_Z_value, status_message)
        """
        max_iterations = 100  # Failsafe to prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            # Step 1: Calculate y*
            y_star = self._find_y_star()

            # Step 2: Find entering variable
            entering_idx = self._find_entering_variable_index(y_star) # -1 to correct the index

            # If -1 is returned, all reduced costs are positive. Optimal Solution Found!
            if entering_idx == -1:
                break

            # Step 3: Calculate entering column in current basis (A*) and current RHS (b*)
            A_star = self.B_inv @ self._A[:, entering_idx]
            b_star = self.B_inv @ self._b

            # Step 4: Minimum Ratio Test (Find leaving row)
            leaving_row = self._minimum_ratio_test(A_star, b_star)

            if leaving_row == -1: # Status is set to unbounded in the helper method if min ratio test returns -1
                break

            # Step 5: Update the Inverse Basis (B_inv) using matrix multiplication (Eta & E)
            self.B_inv = self._build_new_B_inv(A_star, leaving_row)

            # Step 6: Update our tracking variables for the new basis
            self.basic_indices[leaving_row] = entering_idx
            self.cB[leaving_row] = self._c[entering_idx]

            iteration += 1

        # Calculate final answers to return
        final_b_star = self.B_inv @ self._b
        final_z = float(self.cB @ final_b_star)

        return final_b_star, final_z, self.status



"""Section for testing engine in the console with a defined problem"""


def test_simplex():
    """
    Tests the SimplexSolver with a predefined 2D LP problem:
    Max Z = 3000x1 + 4000x2
    s.t.
      2x1 +  0x2 <= 8
      0x1 + 3x2 <= 18
     3x1 +  2x2 <= 18
    """
    timeStart = dt.perf_counter()

    # Notice the 2D array structure here
    A = [
        [2.0, 0.0],
        [0.0, 3.0],
        [3.0, 2.0]
    ]
    b = [8.0, 18.0, 18.0]
    c = [3000.0, 4000.0]

    print("--- Starting Simplex Solver Test ---")
    solver = SimplexSolver(obj_coefficients=c, constraint_matrix=A, rhs_values=b, maximize=True)

    final_b_star, max_z, final_status = solver.solve()

    print(f"Final Status: {final_status}")
    print(f"Optimal Z Value: {max_z}")
    print("Basic Variables in solution:")
    for row_idx, var_idx in enumerate(solver.basic_indices):
        print(f"  x{var_idx + 1} = {round(final_b_star[row_idx], 3)}")

    print(f"Calculation finished in {((dt.perf_counter()) - timeStart)*1000:.2f} milliseconds.")


# Run the test if this file is executed directly
if __name__ == "__main__":
    test_simplex()