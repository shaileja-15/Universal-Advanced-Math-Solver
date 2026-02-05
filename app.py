import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from openai import OpenAI

# -------------------
# CONFIG
# -------------------
client = OpenAI(api_key="YOUR_API_KEY")

transformations = standard_transformations + (implicit_multiplication_application,)
x = sp.symbols('x')

st.set_page_config("Math Generative AI", "🧠", layout="wide")
st.title("🧠 Mathematical Generative AI Solver")

question = st.text_area("Enter any math question")

mode = st.selectbox("Solve Mode", [
    "AI Explain + Solve",
    "Symbolic Solve"
])

run = st.button("Solve")

# -------------------
# SYMBOLIC ENGINE
# -------------------
def symbolic_try(q):
    try:
        expr = parse_expr(q.replace("^","**"), transformations=transformations)
        return sp.solve(expr, x)
    except:
        return None

# -------------------
# RUN
# -------------------
if run and question:

    if mode == "Symbolic Solve":
        res = symbolic_try(question)
        st.write(res)

    else:

        prompt = f"""
Solve this math problem step by step and explain clearly:

{question}
"""

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )

        st.markdown("### 🤖 AI Solution")
        st.write(resp.choices[0].message.content)
