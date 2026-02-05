import streamlit as st
import sympy as sp
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Math Gen-AI", page_icon="🧮", layout="wide")

# --- API SETUP ---
# Replace with your actual API key or use st.secrets for deployment
API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def solve_math_query(query):
    """Uses LLM to translate natural language to SymPy commands."""
    prompt = f"""
    Translate the following math problem into a Python dictionary with two keys:
    'type': (either 'solve', 'derivative', 'integrate', or 'simplify')
    'expr': (the mathematical expression in string format, using ** for power)
    
    Problem: "{query}"
    Return ONLY a valid Python dictionary. 
    Example: {{"type": "solve", "expr": "x**2 - 4"}}
    """
    response = model.generate_content(prompt)
    # Basic cleaning of the LLM response
    data = eval(response.text.strip().replace("```python", "").replace("```", ""))
    return data

# --- UI LAYOUT ---
st.title("🧮 Mathematical Generative AI")
st.markdown("Enter a math problem in plain English (e.g., *'What is the integral of x squared?'* or *'Solve x^2 + 5x + 6 = 0'*).")

query = st.text_input("Your Mathematical Question:", placeholder="e.g. Find the derivative of sin(x)*exp(x)")

if query:
    try:
        with st.spinner("Calculating..."):
            # 1. Parse logic using AI
            parsed = solve_math_query(query)
            q_type = parsed['type']
            q_expr = parsed['expr']
            
            # 2. Define Symbols
            x = sp.Symbol('x')
            expr = sp.sympify(q_expr)
            
            # 3. Execute Math
            if q_type == "solve":
                result = sp.solve(expr, x)
                label = "Solution(s) for x"
            elif q_type == "derivative":
                result = sp.diff(expr, x)
                label = "Derivative"
            elif q_type == "integrate":
                result = sp.integrate(expr, x)
                label = "Indefinite Integral"
            else:
                result = sp.simplify(expr)
                label = "Simplified Expression"

            # --- DISPLAY RESULTS ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Results")
                st.write(f"**Action:** {q_type.capitalize()}")
                st.latex(f"f(x) = {sp.latex(expr)}")
                st.write(f"**{label}:**")
                st.latex(sp.latex(result))

            with col2:
                st.subheader("Visualization")
                # Create plot
                f_num = sp.lambdify(x, expr, 'numpy')
                x_vals = np.linspace(-10, 10, 400)
                try:
                    y_vals = f_num(x_vals)
                    fig, ax = plt.subplots()
                    ax.plot(x_vals, y_vals, label="Original Function", color="#00f2ff")
                    ax.axhline(0, color='black', lw=1)
                    ax.axvline(0, color='black', lw=1)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.legend()
                    st.pyplot(fig)
                except:
                    st.warning("Could not generate a plot for this specific expression.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Tip: Try to be specific, like 'Solve for x: x**2 = 16'")

# --- FOOTER ---
st.sidebar.info("This AI uses SymPy for symbolic accuracy and Gemini for natural language parsing.")
