import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import pandas as pd

def sair_model(t, y, alpha, beta_SI, beta_AI, sigma_IS, sigma_RS, delta):
    S, A, I, R = y
    
    dSdt = -alpha * S * A - beta_SI * S * I + sigma_IS * I + sigma_RS * R
    dAdt = alpha * S * A - beta_AI * A * I
    dIdt = beta_SI * S * I + beta_AI * A * I - sigma_IS * I - delta * I
    dRdt = delta * I - sigma_RS * R
    
    return [dSdt, dAdt, dIdt, dRdt]

def euler_solver(func, t_span, y0, args, num_steps):
    t0, tf = t_span
    h = (tf - t0) / num_steps
    t_points = np.linspace(t0, tf, num_steps + 1)
    y_points = np.zeros((len(y0), num_steps + 1))
    y_points[:, 0] = y0
    
    f = lambda t, y: np.array(func(t, y, *args))
    
    for i in range(num_steps):
        t = t_points[i]
        y = y_points[:, i]
        y_points[:, i+1] = y + h * f(t, y)
        
    return t_points, y_points

def heun_solver(func, t_span, y0, args, num_steps):
    t0, tf = t_span
    h = (tf - t0) / num_steps
    t_points = np.linspace(t0, tf, num_steps + 1)
    y_points = np.zeros((len(y0), num_steps + 1))
    y_points[:, 0] = y0
    
    f = lambda t, y: np.array(func(t, y, *args))
    
    for i in range(num_steps):
        t = t_points[i]
        y = y_points[:, i]
        
        k1 = f(t, y)
        k2 = f(t + h, y + h * k1)
        y_points[:, i+1] = y + (h / 2.0) * (k1 + k2)
        
    return t_points, y_points

def rk4_solver(func, t_span, y0, args, num_steps):
    t0, tf = t_span
    h = (tf - t0) / num_steps
    t_points = np.linspace(t0, tf, num_steps + 1)
    y_points = np.zeros((len(y0), num_steps + 1))
    y_points[:, 0] = y0
    
    f = lambda t, y: np.array(func(t, y, *args))
    
    for i in range(num_steps):
        t = t_points[i]
        y = y_points[:, i]
        
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y_points[:, i+1] = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return t_points, y_points

def calculate_jacobian(S, A, I, R, args):
    alpha, beta_SI, beta_AI, sigma_IS, sigma_RS, delta = args
    
    J = np.zeros((4, 4))
    
    J[0, 0] = -alpha * A - beta_SI * I
    J[0, 1] = -alpha * S
    J[0, 2] = -beta_SI * S + sigma_IS
    J[0, 3] = sigma_RS
    
    J[1, 0] = beta_SI * I
    J[1, 1] = beta_AI * I
    J[1, 2] = beta_SI * S + beta_AI * A - sigma_IS - delta
    J[1, 3] = 0
    
    J[2, 0] = 0
    J[2, 1] = 0
    J[2, 2] = delta
    J[2, 3] = -sigma_RS
    
    J[3, 0] = alpha * A
    J[3, 1] = alpha * S - beta_AI * I
    J[3, 2] = -beta_AI * A
    J[3, 3] = 0
    
    return J

def check_stability(eigenvalues):
    tolerance = 1e-9
    real_parts = np.real(eigenvalues)
    
    if np.all(real_parts <= tolerance):
        if np.any(np.abs(real_parts) < tolerance):
            return "Lyapunov stable (or marginally stable)", "gray"
        else:
            return "asymptotically stable", "green"
    else:
        return "unstable", "red"

st.set_page_config(layout="wide")

st.title("Epidemiological Models Applied to Viruses in Computer Networks")

st.markdown("""
This application simulates the **SAIR (Susceptible-Antidotal-Infected-Removed)** model presented in the 2005 paper by Piqueira, Navarro, and Monteiro.
The goal is to analyze the propagation dynamics of a computer virus in a network, treating it as an epidemic.
The model modifies the traditional SIR model by introducing an "Antidotal" (A) compartment for computers equipped with anti-virus software.
""")

st.header("The SAIR model")
st.markdown("""
The model is described by a system of four Ordinary Differential Equations (ODEs). The simulation assumes a closed network where the influx of new computers (N) and the natural mortality rate (μ) are zero.
""")

st.subheader("Variables")
st.markdown("""
- **S(t):** Susceptible computers (non-infected, no anti-virus)
- **A(t):** Antidotal computers (non-infected, with anti-virus)
- **I(t):** Infected computers
- **R(t):** Removed computers
""")

st.subheader("System of ODEs")
st.latex(r'''
\frac{dS}{dt} = -\alpha S A - \beta_{SI} S I + \sigma_{IS} I + \sigma_{RS} R
''')
st.latex(r'''
\frac{dI}{dt} = \beta_{SI} S I + \beta_{AI} A I - \sigma_{IS} I - \delta I
''')
st.latex(r'''
\frac{dR}{dt} = \delta I - \sigma_{RS} R
''')
st.latex(r'''
\frac{dA}{dt} = \alpha S A - \beta_{AI} A I
''')

st.subheader("Parameters")
st.markdown(r"""
- **$\alpha$**: Conversion rate of Susceptible to Antidotal (e.g., installing anti-virus).
- **$\beta_{SI}$**: Infection rate of Susceptible computers.
- **$\beta_{AI}$**: Infection rate of Antidotal computers (e.g., by a new virus variant).
- **$\sigma_{IS}$**: Recovery rate of Infected computers (becoming Susceptible).
- **$\sigma_{RS}$**: Recovery rate of Removed computers (becoming Susceptible).
- **$\delta$**: Removal rate of Infected computers.
""")


st.sidebar.header("Simulation settings")

st.sidebar.subheader("Model parameters")
alpha = st.sidebar.slider(r"$\alpha$", 0.0, 1.0, 0.1)
beta_SI = st.sidebar.slider(r"$\beta_{SI}$", 0.0, 1.0, 0.5)
beta_AI = st.sidebar.slider(r"$\beta_{AI}$", 0.0, 1.0, 0.01)
sigma_IS = st.sidebar.slider(r"$\sigma_{IS}$", 0.0, 1.0, 0.5)
sigma_RS = st.sidebar.slider(r"$\sigma_{RS}$", 0.0, 1.0, 0.5)
delta = st.sidebar.slider(r"$\delta$", 0.0, 1.0, 0.5)

st.sidebar.subheader("Initial conditions")
S0 = st.sidebar.number_input(r"Initial Susceptible ($S_0$)", min_value=0.0, value=10.0)
A0 = st.sidebar.number_input(r"Initial Antidotal ($A_0$)", min_value=0.0, value=35.0)
I0 = st.sidebar.number_input(r"Initial Infected ($I_0$)", min_value=0.0, value=5.0)
R0 = st.sidebar.number_input(r"Initial Removed ($R_0$)", min_value=0.0, value=0.0)
num_steps = st.sidebar.slider(r"Number of steps (for comparing own with SciPy solvers)", 10, 5000, 100)

st.sidebar.subheader("Time settings")
t_max = st.sidebar.slider(r"Max time ($t_{max}$)", 1, 100, 20)


t_span = (0, t_max)
t_eval = np.linspace(t_span[0], t_span[1], 500)
y0 = [S0, A0, I0, R0]
params = (alpha, beta_SI, beta_AI, sigma_IS, sigma_RS, delta)

sol = solve_ivp(
    sair_model, 
    t_span, 
    y0, 
    args=params, 
    method='RK45', 
    t_eval=t_eval
)

t = sol.t
S, A, I, R = sol.y


st.header("Simulation results")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t, y=A,
    mode='lines',
    name='Antidote (A)',
    line=dict(color='black', width=2)
))

fig.add_trace(go.Scatter(
    x=t, y=S,
    mode='lines',
    name='Susceptible (S)',
    line=dict(color='black', width=2, dash='dash')
))

fig.add_trace(go.Scatter(
    x=t, y=I,
    mode='lines',
    name='Infected (I)',
    line=dict(color='black', width=2, dash='dot')
))

fig.add_trace(go.Scatter(
    x=t, y=R,
    mode='lines',
    name='Removed (R)',
    line=dict(color='black', width=2, dash='dashdot')
))

fig.update_layout(
    title='SAIR Model Dynamics',
    xaxis_title='Time',
    yaxis_title='Population',
    legend_title='Compartments',
    hovermode="x unified",
    height=700
)

st.plotly_chart(fig, use_container_width=True)


st.header("Solver error comparison")
with st.spinner("Running solver comparison..."):
    t_points_manual = np.linspace(t_span[0], t_span[1], num_steps + 1)
    sol_true = solve_ivp(
        sair_model, 
        t_span, 
        y0, 
        args=params, 
        method='DOP853', 
        t_eval=t_points_manual, 
        rtol=1e-12, 
        atol=1e-12
    )
    y_true = sol_true.y
    
    solvers_to_test = ['RK45', 'RK23', 'BDF', 'Radau', 'LSODA']
    results = []
    
    for solver in solvers_to_test:
        sol_test = solve_ivp(
            sair_model, 
            t_span, 
            y0, 
            args=params, 
            method=solver, 
            t_eval=t_points_manual
        )
        y_pred = sol_test.y
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        mae = np.mean(np.abs(y_pred-y_true))
        mape = np.mean(np.abs(y_pred-y_true)) * 100 / np.mean(y_true)
        results.append({'Solver': solver, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

    manual_solvers = {
        "Euler (Order 1)": euler_solver,
        "Heun (Order 2)": heun_solver,
        "RK4 (Order 4)": rk4_solver
    }
    
    for name, solver_func in manual_solvers.items():
        t_pred, y_pred = solver_func(sair_model, t_span, y0, params, num_steps)
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        mae = np.mean(np.abs(y_pred-y_true))
        mape = np.mean(np.abs(y_pred-y_true)) * 100 / np.mean(y_true)
        results.append({'Solver': name, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

    st.dataframe(pd.DataFrame(results))
    st.markdown("RMSE is calculated against 'DOP853' solver solution based on the current parameters.")

st.header("Jacobian and stability analysis")
st.markdown("""
The stability of the system can be analyzed by examining the eigenvalues of the Jacobian matrix evaluated at the system's equilibrium points. This analysis helps determine if the 'disease-free' state is stable (i.e., the virus dies out).
The Jacobian matrix $J$ (with respect to variables $[S, A, I, R]$) is given in the paper as:
""")
st.latex(r'''
J = \begin{bmatrix}
-\alpha A - \beta_{SI} I & -\alpha S & -\beta_{SI} S + \sigma_{IS} & \sigma_{RS} \\
\beta_{SI} I & \beta_{AI} I & \beta_{SI} S + \beta_{AI} A - \sigma_{IS} - \delta & 0 \\
0 & 0 & \delta & -\sigma_{RS} \\
\alpha A & \alpha S - \beta_{AI} I & -\beta_{AI} A & 0
\end{bmatrix}
''')
st.markdown("""
We can analyze the two 'disease-free' equilibrium points (where $I=0$).
The total population is $T = S_0 + A_0 + I_0 + R_0$.
""")

T_pop = S0 + A0 + I0 + R0

st.markdown("---")
st.subheader("Disease-free point $P_1 = (S=0, A=T, I=0, R=0)$")
st.markdown("This point represents a network where all computers are antidotal.")
S_p1, A_p1, I_p1, R_p1 = 0.0, T_pop, 0.0, 0.0
J_p1 = calculate_jacobian(S_p1, A_p1, I_p1, R_p1, params)
eig_p1 = np.linalg.eigvals(J_p1)
stability_p1, color_p1 = check_stability(eig_p1)
st.markdown(f"**Stability:** :{color_p1}[{stability_p1}]")

st.markdown("Numerically calculated eigenvalues:")
st.write([f"{e.real:.2f} + {e.imag:.2f}j" for e in eig_p1])

st.markdown("Theoretical eigenvalues from paper:")
lambda1_p1 = -T_pop * alpha
lambda2_p1 = beta_AI * T_pop - sigma_IS - delta
lambda3_p1 = -sigma_RS
lambda4_p1 = 0.0
st.markdown(f"- $\lambda_1 = -T \cdot \\alpha = {lambda1_p1:.2f}$")
st.markdown(f"- $\lambda_2 = \\beta_{{AI}} \cdot T - \sigma_{{IS}} - \delta = {lambda2_p1:.2f}$")
st.markdown(f"- $\lambda_3 = -\sigma_{{RS}} = {lambda3_p1:.2f}$")
st.markdown(f"- $\lambda_4 = 0 = {lambda4_p1:.2f}$")

st.markdown("---")
st.subheader("Disease-free point $P_2 = (S=T, A=0, I=0, R=0)$")
st.markdown("This point represents a network where all computers are susceptible.")
S_p2, A_p2, I_p2, R_p2 = T_pop, 0.0, 0.0, 0.0
J_p2 = calculate_jacobian(S_p2, A_p2, I_p2, R_p2, params)
eig_p2 = np.linalg.eigvals(J_p2)
stability_p2, color_p2 = check_stability(eig_p2)
st.markdown(f"**Stability:** :{color_p2}[{stability_p2}]")

st.markdown("Numerically calculated eigenvalues:")
st.write([f"{e.real:.2f} + {e.imag:.2f}j" for e in eig_p2])

st.markdown("Theoretical eigenvalues from paper:")
lambda1_p2 = 0.0
lambda2_p2 = beta_SI * T_pop - sigma_IS - delta
lambda3_p2 = -sigma_RS
lambda4_p2 = alpha * T_pop
st.markdown(f"- $\lambda_1 = 0 = {lambda1_p2:.2f}$")
st.markdown(f"- $\lambda_2 = \\beta_{{SI}} \cdot T - \sigma_{{IS}} - \delta = {lambda2_p2:.2f}$")
st.markdown(f"- $\lambda_3 = -\sigma_{{RS}} = {lambda3_p2:.2f}$")
st.markdown(f"- $\lambda_4 = \\alpha \cdot T = {lambda4_p2:.2f}$")
st.markdown("The paper states this point is always unstable because $\lambda_4 > 0$ (assuming $\\alpha, T > 0$).")

st.markdown("---")
st.subheader("Endemic point $P_3 = (S^*, 0, I^*, R^*)$")
st.markdown("This endemic point represents a network with no antidotal computers ($A=0$).")
try:
    S_p3 = (sigma_IS + delta) / beta_SI
    A_p3 = 0.0
    
    if S_p3 >= T_pop or (delta + sigma_RS) == 0:
        raise ValueError("Point not physically valid (S >= T or division by zero)")
    
    I_p3 = sigma_RS * (T_pop - S_p3) / (delta + sigma_RS)
    R_p3 = delta * I_p3 / sigma_RS
    
    if I_p3 < 0 or R_p3 < 0:
        raise ValueError("Point not physically valid (negative populations)")
        
    st.markdown(f"Calculated Point: $S={S_p3:.2f}$, $A={A_p3:.2f}$, $I={I_p3:.2f}$, $R={R_p3:.2f}$")
    J_p3 = calculate_jacobian(S_p3, A_p3, I_p3, R_p3, params)
    eig_p3 = np.linalg.eigvals(J_p3)
    stability_p3, color_p3 = check_stability(eig_p3)
    st.markdown(f"**Stability:** :{color_p3}[{stability_p3}]")
    st.markdown("Numerically calculated eigenvalues:")
    st.write([f"{e.real:.2f} + {e.imag:.2f}j" for e in eig_p3])

except (ValueError, ZeroDivisionError) as e:
    st.markdown(f":gray[This endemic point is not valid for the current parameters.] (Reason: {e})")

st.markdown("---")
st.subheader("Endemic point $P_4 = (S^*, A^*, I^*, R^*)$")
st.markdown("This endemic point represents a network where all four compartments are non-zero.")

try:
    T_crit = (sigma_IS + delta) / beta_AI
    
    if T_pop <= T_crit:
        raise ValueError(f"Point not valid. $T \le T_{{crit}}$ ($T={T_pop:.2f}, T_{{crit}}={T_crit:.2f}$)")
    
    denominator = (beta_AI - beta_SI) / alpha + (sigma_RS + delta) / sigma_RS
    if denominator == 0:
        raise ZeroDivisionError("Denominator is zero.")
        
    I_p4 = (T_pop - T_crit) / denominator
    
    if I_p4 <= 0:
         raise ValueError("Point not valid ($I \le 0$)")
         
    S_p4 = beta_AI * I_p4 / alpha
    A_p4 = (sigma_IS + delta) / beta_AI - (beta_SI * I_p4 / alpha)
    R_p4 = delta * I_p4 / sigma_RS
    
    if S_p4 < 0 or A_p4 < 0 or R_p4 < 0:
        raise ValueError("Point not physically valid (negative populations)")
    
    st.markdown(f"Calculated point: $S={S_p4:.2f}$, $A={A_p4:.2f}$, $I={I_p4:.2f}$, $R={R_p4:.2f}$")
    J_p4 = calculate_jacobian(S_p4, A_p4, I_p4, R_p4, params)
    eig_p4 = np.linalg.eigvals(J_p4)
    stability_p4, color_p4 = check_stability(eig_p4)
    st.markdown(f"**Stability:** :{color_p4}[{stability_p4}]")
    st.markdown("Numerically calculated eigenvalues:")
    st.write([f"{e.real:.2f} + {e.imag:.2f}j" for e in eig_p4])

except (ValueError, ZeroDivisionError) as e:
    st.markdown(f":gray[This endemic point is not valid for the current parameters.] (Reason: {e})")