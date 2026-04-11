import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------
# NUMERICAL INTEGRATION METHODS
# --------------------------------------------------------
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    result = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return result, x, y

def midpoint_rule(f, a, b, n):
    h = (b - a) / n
    m = a + (np.arange(n) + 0.5) * h
    y = f(m)
    result = h * np.sum(y)
    return result, m, y

def simpsons_rule(f, a, b, n):
    if n % 2 != 0:
        n += 1  # Force n to be even
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    ends = y[0] + y[-1]
    odds = 4 * np.sum(y[1:-1:2])
    evens = 2 * np.sum(y[2:-1:2])
    result = (h / 3) * (ends + odds + evens)
    return result, x, y

def romberg_integration(f, a, b, n, initial_intervals=1):
    R = np.zeros((n, n))
    for i in range(n):
        intervals = initial_intervals * (2**i)
        R[i, 0], _, _ = trapezoidal_rule(f, a, b, intervals)
        
    for j in range(1, n):
        for i in range(j, n):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
            
    return R

def double_integral_midpoint(f, a, b, c, d, nx, ny):
    hx = (b - a) / nx
    hy = (d - c) / ny
    xi = a + (np.arange(nx) + 0.5) * hx
    yj = c + (np.arange(ny) + 0.5) * hy
    X, Y = np.meshgrid(xi, yj)
    Z = f(X, Y)
    result = hx * hy * np.sum(Z)
    return result, X, Y, Z

# --------------------------------------------------------
# HELPER FOR MATH EVALUATION
# --------------------------------------------------------
def get_eval_env(**kwargs):
    """Creates a safe evaluation environment with common math functions."""
    env = {"np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos,
           "tan": np.tan, "log": np.log, "sqrt": np.sqrt, "pi": np.pi, "e": np.e}
    env.update(kwargs)
    return env

# --------------------------------------------------------
# STREAMLIT APP LAYOUT
# --------------------------------------------------------
st.set_page_config(page_title="Numerical Integration Applet", layout="wide")
st.title("Interactive Numerical Integration")
st.markdown("*By Dr Nouralden Mohammed*")

tab1, tab2, tab3 = st.tabs(["1D Integration Rules", "Romberg Integration", "Double Integrals"])

# --- TAB 1: 1D INTEGRATION RULES ---
with tab1:
    st.header("Standard 1D Integration Methods")
    st.markdown("Compare Trapezoidal, Midpoint, and Simpson's Rules for $\int_a^b f(x) dx$.")
    
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.markdown("**Function Definition:**")
        func_expr = st.text_input("f(x):", value="1 / (1 + x**2)")
        
        st.markdown("**Parameters:**")
        c1, c2 = st.columns(2)
        with c1:
            a_val = st.number_input("Lower limit (a)", value=0.0)
        with c2:
            b_val = st.number_input("Upper limit (b)", value=1.0)
            
        n_val = st.slider("Subintervals (n)", 2, 100, 6)
        
        methods_selected = st.multiselect(
            "Select Methods to Compare:",
            ["Trapezoidal", "Midpoint", "Simpson's"],
            default=["Trapezoidal", "Simpson's"]
        )
        
    with col2:
        try:
            def f_custom(x):
                return eval(func_expr, get_eval_env(x=x))
            
            # Exact evaluation via SciPy
            exact_val, _ = spi.quad(f_custom, a_val, b_val)
            st.markdown(f"**Exact Integral (SciPy):** `{exact_val:.8f}`")
            
            # Plotting Setup
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            x_dense = np.linspace(a_val, b_val, 500)
            y_dense = f_custom(x_dense)
            
            ax1.plot(x_dense, y_dense, 'k-', linewidth=2, label="Exact f(x)")
            ax1.fill_between(x_dense, y_dense, alpha=0.1, color='gray')
            
            results_data = []
            
            # Calculate & Plot based on selections
            if "Trapezoidal" in methods_selected:
                res_trap, x_trap, y_trap = trapezoidal_rule(f_custom, a_val, b_val, n_val)
                err_trap = abs(exact_val - res_trap)
                results_data.append({"Method": "Trapezoidal", "Result": res_trap, "Error": err_trap})
                ax1.plot(x_trap, y_trap, 'o--', color='blue', alpha=0.7, label='Trapezoidal Nodes')
                
            if "Midpoint" in methods_selected:
                res_mid, x_mid, y_mid = midpoint_rule(f_custom, a_val, b_val, n_val)
                err_mid = abs(exact_val - res_mid)
                results_data.append({"Method": "Midpoint", "Result": res_mid, "Error": err_mid})
                ax1.plot(x_mid, y_mid, 's', color='green', alpha=0.7, label='Midpoints')
                # Optionally plot bars to visualize midpoint rectangles
                ax1.bar(x_mid, y_mid, width=(b_val-a_val)/n_val, alpha=0.2, color='green', edgecolor='green')
                
            if "Simpson's" in methods_selected:
                n_simp = n_val if n_val % 2 == 0 else n_val + 1
                res_simp, x_simp, y_simp = simpsons_rule(f_custom, a_val, b_val, n_simp)
                err_simp = abs(exact_val - res_simp)
                results_data.append({"Method": f"Simpson's (n={n_simp})", "Result": res_simp, "Error": err_simp})
                ax1.plot(x_simp, y_simp, '^:', color='red', alpha=0.7, label="Simpson's Nodes")

            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.set_title(f"Numerical Integration of f(x) from {a_val} to {b_val}")
            ax1.legend()
            ax1.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig1)
            
            # Display results
            if results_data:
                st.table(pd.DataFrame(results_data).set_index("Method").style.format({"Result": "{:.6f}", "Error": "{:.2e}"}))
                
        except Exception as e:
            st.error(f"Error evaluating mathematical expression: {e}")

# --- TAB 2: ROMBERG INTEGRATION ---
with tab2:
    st.header("Romberg Integration")
    st.markdown("Generates a table of approximations using Richardson Extrapolation to eliminate error terms.")
    
    col3, col4 = st.columns([1, 2.5])
    with col3:
        func_expr_romb = st.text_input("f(x) for Romberg:", value="1 / (1 + x**2)", key="romb_f")
        c3, c4 = st.columns(2)
        with c3:
            a_romb = st.number_input("Lower limit (a)", value=0.0, key="romb_a")
        with c4:
            b_romb = st.number_input("Upper limit (b)", value=1.0, key="romb_b")
            
        n_rows = st.slider("Number of Rows (n)", 2, 8, 4)
        init_intervals = st.number_input("Initial Intervals", value=1, min_value=1)
        
    with col4:
        try:
            def f_romb(x):
                return eval(func_expr_romb, get_eval_env(x=x))
                
            exact_romb, _ = spi.quad(f_romb, a_romb, b_romb)
            st.markdown(f"**Exact Integral (SciPy):** `{exact_romb:.8f}`")
            
            # Compute Romberg Table
            R = romberg_integration(f_romb, a_romb, b_romb, n_rows, init_intervals)
            
            # Format table nicely
            columns = [f"O(h^{2*(j+1)})" for j in range(n_rows)]
            index = [f"k={i} (n={init_intervals * 2**i})" for i in range(n_rows)]
            
            df_R = pd.DataFrame(R, columns=columns, index=index)
            # Replace zeros in upper triangle with empty string for better readability
            df_R = df_R.map(lambda x: f"{x:.8f}" if x != 0 else "")
            
            st.markdown("**Romberg Table:**")
            st.dataframe(df_R, use_container_width=True)
            
            best_est = R[-1, -1]
            st.success(f"**Best Estimate:** `{best_est:.8f}`  |  **Absolute Error:** `{abs(exact_romb - best_est):.2e}`")
            
        except Exception as e:
            st.error(f"Error evaluating mathematical expression: {e}")

# --- TAB 3: DOUBLE INTEGRALS ---
with tab3:
    st.header("Double Integration (2D Midpoint Rule)")
    st.markdown("Approximates the volume under the surface $z = f(x, y)$ over the region $[a, b] \\times [c, d]$.")
    
    col5, col6 = st.columns([1, 2.5])
    with col5:
        st.markdown("**Function Definition:**")
        func_2d_expr = st.text_input("f(x, y):", value="x**2 + y**2")
        
        st.markdown("**X Boundaries:**")
        c5, c6 = st.columns(2)
        with c5: a_2d = st.number_input("Start x (a)", value=0.0)
        with c6: b_2d = st.number_input("End x (b)", value=1.0)
        
        st.markdown("**Y Boundaries:**")
        c7, c8 = st.columns(2)
        with c7: c_2d = st.number_input("Start y (c)", value=0.0)
        with c8: d_2d = st.number_input("End y (d)", value=1.0)
        
        st.markdown("**Grid Resolution:**")
        nx_val = st.slider("Intervals in x (nx)", 2, 50, 10)
        ny_val = st.slider("Intervals in y (ny)", 2, 50, 10)
        
    with col6:
        try:
            def f_2d(x, y):
                return eval(func_2d_expr, get_eval_env(x=x, y=y))
            
            # SciPy dblquad expects function as f(y, x), limits for y are functions of x
            exact_2d, _ = spi.dblquad(lambda y, x: f_2d(x, y), a_2d, b_2d, lambda x: c_2d, lambda x: d_2d)
            
            res_2d, X, Y, Z = double_integral_midpoint(f_2d, a_2d, b_2d, c_2d, d_2d, nx_val, ny_val)
            err_2d = abs(exact_2d - res_2d)
            
            st.metric(label="Approximate Volume", value=f"{res_2d:.6f}", delta=f"Error: {err_2d:.2e}", delta_color="inverse")
            st.markdown(f"**Analytical/Exact Volume (SciPy):** `{exact_2d:.6f}`")
            
            # 3D Plot
            fig3 = plt.figure(figsize=(10, 6))
            ax3 = fig3.add_subplot(111, projection='3d')
            
            # Plot surface
            surf = ax3.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
            fig3.colorbar(surf, ax=ax3, shrink=0.5, aspect=10)
            
            ax3.set_xlabel('X axis')
            ax3.set_ylabel('Y axis')
            ax3.set_zlabel('Z = f(x, y)')
            ax3.set_title("Function Surface (Evaluated at Midpoints)")
            
            st.pyplot(fig3)
            
        except Exception as e:
            st.error(f"Error evaluating mathematical expression: {e}")
