from ctypes import windll
import tkinter as tk
import time as dt
from tkinter import ttk, messagebox

import numpy as np
import matplotlib.pyplot as plt

from core.simplex_engine import SimplexSolver
#from core.visualizer import plot_feasible_region

class SimplexGUIMain:
    def __init__(self, root:tk.Tk):
        self.root = root
        self.root.title("2D & 3D Simplex LP Solver by Miguel Camarena")
        self.root.geometry("800x600")

        # we want to lay out the same variables from our simplex solver's constructor
        # calculation settings
        self.num_decision_variables = tk.IntVar(value=2)
        self.num_constraints = tk.IntVar(value=3)
        self.maximize = tk.BooleanVar(value=True)

        # makes lists that point to the entry boxes that hold our necessary values
        self.obj_func_entries: list[tk.Entry] = [] # makes c vector
        self.constraint_entries: list[list[tk.Entry]] = [] # makes A matrix
        self.rhs_entries: list[tk.Entry] = [] # makes b vector

        # --- UI Frame Initialization ---
        # We split the window into distinct panels (Top, Middle, Bottom)
        self.settings_frame = ttk.LabelFrame(self.root, text="Settings")
        self.settings_frame.pack(fill="x", padx=10, pady=30)

        self.input_frame = ttk.LabelFrame(self.root, text="Problem Formulation")
        self.input_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_frame = ttk.LabelFrame(self.root, text="Results")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Build the initial UI components
        self._build_settings()
        self._build_input_grid()
        self._build_results_panel()


    # UI CONSTRUCTION METHODS

    def _build_settings(self) -> None:
        """Builds the radio buttons and spinboxes for problem dimensions."""

        self._dv_setting2D = (ttk.Radiobutton(self.settings_frame, text="2D", variable=self.num_decision_variables, value=2))
        self._dv_setting2D.pack(side="left", padx=25, pady=5)

        self._dv_setting3D = (ttk.Radiobutton(self.settings_frame, text="3D", variable=self.num_decision_variables, value=3))
        self._dv_setting3D.pack(side="left", padx=25, pady=5)

        self._maximize =     ttk.Checkbutton(self.settings_frame, text="Maximize", variable=self.maximize)
        self._maximize.pack(side="left", padx=25, pady=5)

        self._generate_grid = ttk.Button(self.settings_frame, text="Generate grid", command=self._build_input_grid)
        self._generate_grid.pack(side="left", padx=25, pady=5,ipadx=5)

        (ttk.Label(self.settings_frame, text="Set # of Constraints:")
         .pack(side="left", padx=25, pady=5,ipadx=5))

        self._set_constraints = ttk.Spinbox(self.settings_frame, textvariable=self.num_constraints)
        self._set_constraints.pack(side="left", padx=10, pady=5,ipadx=5)


    def _build_input_grid(self) -> None:
        """Clears the current input frame and dynamically generates text boxes."""
        # 1. Clear existing widgets in the frame (so they don't stack)
        for widget in self.input_frame.winfo_children():
            widget.destroy()

        # 2. Clear your state lists
        self.obj_func_entries.clear()
        self.constraint_entries.clear()
        self.rhs_entries.clear()

        # 3. Read current dimensions
        vars_count = self.num_decision_variables.get()
        cons_count = self.num_constraints.get()

        widthSize = 5

        #text that says Z =
        (ttk.Label(self.input_frame, text="Z =", font=("Arial", 10, "bold"))
            .grid(row=0, column=0, padx=(10, 2), pady=5,sticky="e"))

        cur_col = 1
        # Use a standard 'for' loop to create the Objective Function (c) tk.Entry boxes
        for j in range(vars_count):
            # 1. Create the Entry Box
            entry = ttk.Entry(self.input_frame, width=widthSize)
            entry.grid(row=0, column=cur_col, padx=2, pady=5)
            self.obj_func_entries.append(entry)
            cur_col += 1

            # 2. Create the trailing Label (e.g., "x1 +" or just "x2")
            if j < vars_count - 1:
                label_text = f"x{j + 1} +"
            else:
                label_text = f"x{j + 1}"  # The last variable doesn't have a '+' after it

            ttk.Label(self.input_frame, text=label_text).grid(row=0, column=cur_col, padx=2, pady=5, sticky="w")
            cur_col += 1


        # nested for loops (rows = cons_count, cols = vars_count) to create
        # the Constraint Matrix (A) tk.Entry boxes, and the RHS (b) tk.Entry boxes.

         #increase constraint count and align A and b under the c row
        for i in range(cons_count):
            row_entries = []

            cur_col = 1
            # we create a list of entry boxes and pack them into a grid
            for j in range(vars_count):
                # 1. Create the Entry Box (Notice how .grid() is on a separate line to prevent the NoneType bug!)
                entry = ttk.Entry(self.input_frame, width=widthSize)
                entry.grid(row=i + 1, column=cur_col, padx=2, pady=2)
                row_entries.append(entry)
                cur_col += 1

                # 2. Create the trailing Label (e.g., "x1 +" or "x2 <=")
                if j < vars_count - 1:
                    label_text = f"x{j + 1} +"
                else:
                    label_text = f"x{j + 1} <="  # The last variable gets the inequality symbol

                (ttk.Label(self.input_frame, text=label_text)
                 .grid(row=i + 1, column=cur_col, padx=2, pady=2,sticky="w"))
                cur_col += 1

            #pack b column after creating the A matrix for tab order continuity
            b_entry = ttk.Entry(self.input_frame, width=widthSize)
            b_entry.grid(row=i + 1, column=cur_col, padx=2, pady=2)
            self.rhs_entries.append(b_entry)

            self.constraint_entries.append(row_entries)

    def _build_results_panel(self) -> None:
        """Builds the solve button and the text area to display answers."""
        solve_btn = ttk.Button(self.results_frame, text="Solve!", command=self._on_solve_clicked)
        solve_btn.pack(pady=5)

        # A text widget to write our final answers into
        self.result_text = tk.Text(self.results_frame, height=10, state="disabled")
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)


    def _on_solve_clicked(self) -> None:
        """Triggered when the user clicks Solve."""
        try:
            process_start = dt.perf_counter()
            # 1. Extract data (You will implement this below)
            c, A, b = self._extract_data_from_ui()

            # 2. Instantiate your engine
            solver = SimplexSolver(
                obj_coefficients=c,
                constraint_matrix=A,
                rhs_values=b,
                maximize=self.maximize.get()
            )

            # 3. Run the math
            final_b_star, max_z, status = solver.solve()

            # 4. Display the results
            self._display_results(final_b_star, max_z, status, solver.basic_indices, process_start)

        except ValueError:
            # This catches errors if the user left a box blank or typed a letter
            messagebox.showerror("Input Error", "Please ensure all fields contain valid numbers.")
        except Exception as e:
            # Catches unexpected math or logic errors
            messagebox.showerror("Solver Error", f"An error occurred: {str(e)}")

    def _extract_data_from_ui(self) -> tuple[list[float], list[list[float]], list[float]]:
        """Reads strings from tk.Entry widgets and converts them to floats."""
        c = []
        A = []
        b = []

        # Loop through self.obj_entries, call .get(), cast to float(), and append to c
        for dv in self.obj_func_entries:
            c.append(float(dv.get()))

        # Loop through self.constraint_entries to build the 2D list A
        for row in self.constraint_entries:
            row_data = []  # Create a new list for this specific row
            for col in row:
                row_data.append(float(col.get()))
            A.append(row_data)  # Append the row list to the main A matrix

        # Loop through self.rhs_entries to build the 1D list b
        for row in self.rhs_entries:
            b.append(float(row.get()))

        return c, A, b

    def _display_results(self, b_star: np.ndarray, z: float, status: str, basic_indices: list[int], time_to_complete) -> None:
        """Writes the final output to the result_text box."""
        # Enable text box so we can write to it
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)  # Clear previous results

        # Format a nice string with the status, Z value, and basic variables
        final_string = (f"Final Status: {status}\n"
                        f"Optimal Z Value: {z}\n"
                        f"Basic Variables in solution:\n")
        for row_idx, var_idx in enumerate(basic_indices):
            final_string += f"  x{var_idx + 1} = {round(b_star[row_idx], 3)}\n"

        final_string += f"Calculation finished in {((dt.perf_counter()) - time_to_complete) * 1000:.2f} milliseconds."

        self.result_text.insert(tk.END, final_string)

        # Lock it again so the user can't type in it
        self.result_text.config(state="disabled")


# --- Application Startup ---
if __name__ == "__main__":
    windll.shcore.SetProcessDpiAwareness(1)

    app_window = tk.Tk()
    app = SimplexGUIMain(app_window)
    app_window.mainloop()
