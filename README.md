# SAIR Model Simulation: Computer Virus Epidemiology

This repository contains an interactive web application built with Streamlit that simulates the SAIR (Susceptible-Antidotal-Infected-Removed) model. This model applies epidemiological concepts to the propagation of viruses within computer networks, based on the work of Piqueira, Navarro, and Monteiro (2005).

## Project Overview

The application solves a system of Ordinary Differential Equations (ODEs) to model the dynamics of computer virus transmission. Unlike the traditional SIR model, the SAIR model introduces an "Antidotal" compartment, representing nodes equipped with antivirus software.

This tool is designed for educational and analytical purposes, allowing users to:
1. Visualize population dynamics over time.
2. Compare the accuracy of different numerical integration methods.
3. Analyze the stability of equilibrium points using Jacobian matrices.

## Features

* **Interactive Simulation:** Adjust model parameters (infection rates, recovery rates, etc.) and initial conditions in real-time via the sidebar.
* **Dynamic Visualization:** View interactive time-series plots of the S, A, I, and R populations using Plotly.
* **Numerical Analysis:** Compares custom implementations of Euler (1st order), Heun (2nd order), and Runge-Kutta (4th order) methods against SciPy's solvers and calculates error metrics (RMSE, MAE, MAPE) for the manual solvers against best-performing SciPy solver.
* **Stability Analysis:** Automatically computes the Jacobian matrix for disease-free and endemic equilibrium points to determine system stability (Lyapunov stable, asymptotically stable, or unstable).

## Mathematical Model

The system is defined by the following differential equations:

$$\frac{dS}{dt} = -\alpha S A - \beta_{SI} S I + \sigma_{IS} I + \sigma_{RS} R$$

$$\frac{dA}{dt} = \alpha S A - \beta_{AI} A I$$

$$\frac{dI}{dt} = \beta_{SI} S I + \beta_{AI} A I - \sigma_{IS} I - \delta I$$

$$\frac{dR}{dt} = \delta I - \sigma_{RS} R$$

Where:
* **S:** Susceptible (vulnerable, no antivirus)
* **A:** Antidotal (protected by antivirus)
* **I:** Infected (active virus)
* **R:** Removed (cleaned or isolated)

## Installation

To run this project locally, ensure you have Python installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/hsannav/simulator-epidemic-computer-virus.git
   cd simulator-epidemic-computer-virus
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install streamlit numpy scipy plotly pandas
   ```

## Usage

Execute the application using Streamlit:

```bash
streamlit run app.py
```

The application will open automatically in your default web browser (usually at `http://localhost:8501`).

## File Structure

* `app.py`: The main application script containing the model logic, solvers, and UI definitions.
* `README.md`: Project documentation.

## Authors

* Fernando Blanco
* Hugo Sánchez

## References

Piqueira, J. R. C., Navarro, B. F., & Monteiro, L. H. A. (2005). Epidemiological models applied to viruses in computer networks. *Journal of Computer and System Sciences*.
