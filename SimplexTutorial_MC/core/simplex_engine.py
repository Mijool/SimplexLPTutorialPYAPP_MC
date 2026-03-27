"""
Core mathematical engine for the Simplex Algorithm.
"""

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
        self._A = np.array(constraint_matrix,
                          dtype=float)
        self._c = np.array(obj_coefficients,
                          dtype=float)
        self._b = np.array(rhs_values,
                          dtype=float)
        self._maximize = maximize

        # initialize problem dimensions (for now only 2)
            # .shape returns a tuple of the dimensions of an array: [rows, cols]
        self._num_constraints: int = self._A.shape[0] # of rows aka constraints
        self._num_vars: int = self._A.shape[1] # of columns aka decision variables

        # setup dynamic variables (The Basis)
        # B_inv should always be an (m x m) matrix, m == # of constraints
        self.B_inv = np.eye(self._num_constraints, dtype=float) # (N= rows, M= cols [set to n as default])
        self.basic_indices: list[int] = []
        self.cB: np.ndarray = np.zeros(self._num_constraints, dtype=float) #coefficient of basic variables

        self.status: str = "Setup Complete"

    def _find_y_star(self) -> np.ndarray:
        """here we find y* (cB*B_inv), we pass our simplex problem in as the parameter (non-static method)"""
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
    def _minimum_ratio_test(self, entering_col: list[float]) -> int:
        """calculate A* = S* @ A and b* = S* @ b to find our leaving variable, return -1 if unbounded (all are negative or undefined)"""
        A_star: np.ndarray = (self.B_inv @ self._A)[:, entering_col] # slices matrix to only get the column we need for our min. ratio test, ':' selects all rows, and we only want the entering variables column

        b_star: np.ndarray = self.B_inv @ self._b

        ratio_arr = (b_star / A_star)
        positive_leaving_indices = np.argwhere( ratio_arr > 0).flatten()

        if positive_leaving_indices.size == 0:
            self.status = "No leaving variable, problem is unbounded"
            return -1
        else:
            leaving_variable_index: int = positive_leaving_indices[ratio_arr[ratio_arr > 0].argmin()]  # we can easily divide matrices in this manner, we return the smallest value that isn't
            self.status = "Leaving variable index is "+str(leaving_variable_index)
            return leaving_variable_index

    def _build_new_B_inv (self) -> np.ndarray:
        """build eta and E and returns E*B_inv for new B_inv"""
        pass
    def solve(self) -> tuple[np.ndarray, str]:
        """solves simplex problem, returns the final z* row and value"""
        pass