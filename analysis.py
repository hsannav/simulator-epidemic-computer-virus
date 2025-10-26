import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def print_introduction():
    print("--- Problem Explanation ---")
    print("This script analyzes the propagation of computer viruses using an epidemiological model.")
    print("The analysis is based on the 2005 paper 'Epidemiological Models Applied to Viruses in Computer Networks'[cite: 1, 4].")
    print("The paper investigates using classical epidemiological models to study computer virus propagation[cite: 8].")
    print("It modifies the standard SIR (Susceptible-Infected-Removed) model by introducing an 'Antidotal' (A) compartment for nodes with anti-virus programs[cite: 9, 32].")
    print("The goal is to analyze this 'SAIR' model to find conditions that prevent the spread of infection in a computer network[cite: 10, 28].")
    print("\n")

def print_model_details():
    print("--- SAIR Model: ODE System, Variables, and Parameters ---")
    print("\nThe model is a system of Ordinary Differential Equations (ODEs) based on the simplified equations (5-8) from the paper, which assume no new computers are added (N=0) and no mortality rate (mu=0) during the fast propagation [cite: 67-70, 91-92].")
    
    print("\nVariables (Compartments):")
    print("S(t): Susceptible computers (non-infected, no anti-virus) [cite: 35]")
    print("A(t): Antidotal computers (non-infected, with anti-virus) [cite: 36]")
    print("I(t): Infected computers [cite: 36]")
    print("R(t): Removed computers [cite: 37]")
    
    print("\nODEs:")
    print("dS/dt = -alpha * S * A - beta_SI * S * I + sigma_IS * I + sigma_RS * R [cite: 67]")
    print("dI/dt = beta_SI * S * I + beta_AI * A * I - sigma_IS * I - delta * I [cite: 69]")
    print("dR/dt = delta * I - sigma_RS * R [cite: 69]")
    print("dA/dt = alpha * S * A - beta_AI * A * I [cite: 70]")
    
    print("\nParameters:")
    print("alpha: Conversion rate of Susceptible to Antidotal [cite: 89]")
    print("beta_SI: Infection rate of Susceptible computers [cite: 62]")
    print("beta_AI: Infection rate of Antidotal computers (e.g., by a new virus) [cite: 82]")
    print("sigma_IS: Recovery rate of Infected computers (becoming Susceptible) [cite: 87]")
    print("sigma_RS: Recovery rate of Removed computers (becoming Susceptible) [cite: 88]")
    print("delta: Removal rate of Infected computers [cite: 84]")
    print("\n")

def sair_model(t, y, alpha, beta_SI, beta_AI, sigma_IS, sigma_RS, delta):
    S, A, I, R = y
    
    dSdt = -alpha * S * A - beta_SI * S * I + sigma_IS * I + sigma_RS * R
    dAdt = alpha * S * A - beta_AI * A * I
    dIdt = beta_SI * S * I + beta_AI * A * I - sigma_IS * I - delta * I
    dRdt = delta * I - sigma_RS * R
    
    return [dSdt, dAdt, dIdt, dRdt]

def run_simulations_and_compare():
    print("--- Solver Comparison ---")
    
    alpha = 0.1
    beta_SI = 0.5
    sigma_IS = 0.5
    sigma_RS = 0.5
    delta = 0.5
    beta_AI = 0.01 
    
    S0 = 10.0
    A0 = 35.0
    I0 = 5.0
    R0 = 0.0
    y0 = [S0, A0, I0, R0]
    
    t_span = (0, 20)
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    
    params = (alpha, beta_SI, beta_AI, sigma_IS, sigma_RS, delta)
    
    sol_true = solve_ivp(
        sair_model, 
        t_span, 
        y0, 
        args=params, 
        method='DOP853', 
        t_eval=t_eval, 
        rtol=1e-12, 
        atol=1e-12
    )
    y_true = sol_true.y
    
    solvers_to_test = ['RK45', 'BDF', 'LSODA']
    results = {}
    
    print("Running simulations to generate ground truth and test solvers...")
    
    for solver in solvers_to_test:
        sol = solve_ivp(
            sair_model, 
            t_span, 
            y0, 
            args=params, 
            method=solver, 
            t_eval=t_eval
        )
        y_pred = sol.y
        
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        results[solver] = rmse
        
    print("\nError Comparison (RMSE against high-accuracy 'DOP853' solution):")
    print("---------------------------------")
    print(f"| {'Solver':<10} | {'RMSE':<18} |")
    print("---------------------------------")
    for solver, rmse in results.items():
        print(f"| {solver:<10} | {rmse:<18.10e} |")
    print("---------------------------------")
    print("\n")
    
    sol_plot = solve_ivp(
        sair_model, 
        t_span, 
        y0, 
        args=params, 
        method='RK45', 
        t_eval=t_eval
    )
    return sol_plot.t, sol_plot.y

def plot_results(t, y, t_span=(0, 20)):
    print("--- Plotting Simulation Results ---")
    
    S, A, I, R = y
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, A, label='Antidote (A)', linestyle='-', color='black')
    plt.plot(t, S, label='Susceptible (S)', linestyle='--', color='black')
    plt.plot(t, I, label='Infected (I)', linestyle=':', color='black')
    plt.plot(t, R, label='Removed (R)', linestyle='-.', marker='*', markevery=20, color='black')
    
    plt.title('SAIR Model Simulation (Asymptotically Stable Disease-Free Case) [cite: 209]')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0, top=60)
    plt.xlim(t_span[0], t_span[1])
    
    print("Displaying plot... Close the plot window to exit the script.")
    plt.show()

def main():
    print_introduction()
    print_model_details()
    
    alpha_param = 0.1
    beta_SI_param = 0.5
    sigma_IS_param = 0.5
    sigma_RS_param = 0.5
    delta_param = 0.5
    beta_AI_param = 0.01
    
    S0_init = 10.0
    A0_init = 35.0
    I0_init = 5.0
    R0_init = 0.0
    y0_init = [S0_init, A0_init, I0_init, R0_init]
    
    t_span = (0, 20)
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    
    model_args = (alpha_param, beta_SI_param, beta_AI_param, sigma_IS_param, sigma_RS_param, delta_param)
    
    sol_true_ref = solve_ivp(
        sair_model, 
        t_span, 
        y0_init, 
        args=model_args, 
        method='DOP853', 
        t_eval=t_eval, 
        rtol=1e-12, 
        atol=1e-12
    )
    y_true_ref = sol_true_ref.y
    
    solvers = ['RK45', 'BDF', 'LSODA']
    solver_results = {}
    
    print("--- Solver Comparison ---")
    print("Running simulations to generate ground truth and test solvers...")

    for s in solvers:
        sol = solve_ivp(
            sair_model, 
            t_span, 
            y0_init, 
            args=model_args, 
            method=s, 
            t_eval=t_eval
        )
        y_pred = sol.y
        rmse = np.sqrt(np.mean((y_pred - y_true_ref)**2))
        solver_results[s] = rmse
            
    print("\nError Comparison (RMSE against high-accuracy 'DOP853' solution):")
    print("---------------------------------")
    print(f"| {'Solver':<10} | {'RMSE':<18} |")
    print("---------------------------------")
    for s, err in solver_results.items():
        print(f"| {s:<10} | {err:<18.10e} |")
    print("---------------------------------")
    print("\n")

    sol_for_plot = solve_ivp(
        sair_model, 
        t_span, 
        y0_init, 
        args=model_args, 
        method='RK45', 
        t_eval=t_eval
    )
    
    plot_results(sol_for_plot.t, sol_for_plot.y, t_span=(0, 20))

if __name__ == "__main__":
    main()