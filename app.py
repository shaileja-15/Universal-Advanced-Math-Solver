import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols
from scipy import optimize

# -------------------------
# CONFIG
# -------------------------
st.set_page_config("Universal Math Solver", "🧠", layout="wide")

x, y, z = sp.symbols('x y z')

st.title("🧠 Universal Mathematics Solver")
st.caption("Algebra • Calculus • Linear Algebra • ODE • Optimization • Graphing")

# -------------------------
# CATEGORY SELECT
# -------------------------
category = st.sidebar.selectbox("Select Math Domain", [
    "Expression Tools",
    "Equation Solver",
    "System of Equations",
    "Calculus",
    "Limits",
    "Series",
    "Matrix Algebra",
    "Differential Equations",
    "Optimization",
    "Graphing",
    "Numeric Root Finder"
])

expr_input = st.text_input("Enter expression")

run = st.button("🚀 Compute")

# -------------------------
# MAIN SOLVER
# -------------------------
if run:

    try:
        st.subheader("Result")

        # ------------------ EXPRESSION ------------------
        if category == "Expression Tools":
            expr = sp.sympify(expr_input)
            tool = st.selectbox("Tool", ["Simplify","Factor","Expand"])
            res = getattr(sp, tool.lower())(expr)
            st.latex(sp.latex(res))

        # ------------------ EQUATION ------------------
        elif category == "Equation Solver":
            expr = sp.sympify(expr_input)
            res = sp.solve(expr, x)
            st.write(res)

        # ------------------ SYSTEM ------------------
        elif category == "System of Equations":
            st.info("Enter eq1, eq2 separated by comma")
            eqs = [sp.sympify(e.strip()) for e in expr_input.split(",")]
            res = sp.solve(eqs, (x,y))
            st.write(res)

        # ------------------ CALCULUS ------------------
        elif category == "Calculus":
            expr = sp.sympify(expr_input)
            op = st.selectbox("Operation", [
                "Derivative","Integral","Partial Derivative"
            ])

            if op == "Derivative":
                order = st.number_input("Order",1,10,1)
                res = sp.diff(expr, x, order)

            elif op == "Integral":
                res = sp.integrate(expr, x)

            else:
                var = st.selectbox("Variable", ["x","y","z"])
                res = sp.diff(expr, symbols(var))

            st.latex(sp.latex(res))

        # ------------------ LIMIT ------------------
        elif category == "Limits":
            expr = sp.sympify(expr_input)
            pt = st.number_input("Limit point", value=0.0)
            res = sp.limit(expr, x, pt)
            st.latex(sp.latex(res))

        # ------------------ SERIES ------------------
        elif category == "Series":
            expr = sp.sympify(expr_input)
            n = st.slider("Order",2,15,6)
            res = sp.series(expr, x, 0, n)
            st.latex(sp.latex(res))

        # ------------------ MATRIX ------------------
        elif category == "Matrix Algebra":
            st.info("Matrix like [[1,2],[3,4]]")
            M = sp.Matrix(sp.sympify(expr_input))
            op = st.selectbox("Matrix Tool", [
                "Determinant","Inverse","Eigenvalues","Rank"
            ])

            if op == "Determinant":
                res = M.det()
            elif op == "Inverse":
                res = M.inv()
            elif op == "Eigenvalues":
                res = M.eigenvals()
            else:
                res = M.rank()

            st.write(res)

        # ------------------ ODE ------------------
        elif category == "Differential Equations":
            st.info("Example: Derivative(y(x),x) - y(x)")
            y_func = sp.Function('y')
            eq = sp.sympify(expr_input)
            res = sp.dsolve(eq, y_func(x))
            st.latex(sp.latex(res))

        # ------------------ OPTIMIZATION ------------------
        elif category == "Optimization":
            expr = sp.sympify(expr_input)
            d = sp.diff(expr, x)
            crit = sp.solve(d, x)
            st.write("Critical Points:", crit)

        # ------------------ GRAPH ------------------
        elif category == "Graphing":
            expr = sp.sympify(expr_input)
            f = sp.lambdify(x, expr, "numpy")
            xs = np.linspace(-10,10,400)
            fig, ax = plt.subplots()
            ax.plot(xs, f(xs))
            ax.axhline(0)
            ax.axvline(0)
            st.pyplot(fig)

        # ------------------ NUMERIC ROOT ------------------
        elif category == "Numeric Root Finder":
            f = sp.lambdify(x, sp.sympify(expr_input), "numpy")
            root = optimize.fsolve(f, 1)
            st.write("Root near guess:", root)

    except Exception as e:
        st.error("Invalid input")
        st.code(str(e))

st.write("---")
st.caption("Powered by SymPy + SciPy + Streamlit")
