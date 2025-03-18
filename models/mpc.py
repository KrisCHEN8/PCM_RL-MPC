import numpy as np
import pandas as pd
import os
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import timedelta
from env.pcm_storage import PCMStorage


system = PCMStorage(dt=900, initial_storage=27.0)


def mpc_controller(system, horizon=4, dt=0.25, datetime='2022-08-31', df=None):
    """
    Model Predictive Controller for the PCM heat pump system.
    """
    # Define the optimization variables
    x = cp.Variable(horizon + 1, nonneg=True)   # SoC of the storage
    u = cp.Variable(horizon, boolean=True)       # Operation mode
    rpm = cp.Variable(horizon, nonneg=True)     # Compressor speed
    Q_dis = cp.Variable(horizon, nonneg=True)   # Discharge heat from PCM in kW
    T_cond = cp.Parameter(horizon)  # Condenser temperature
    e_price = cp.Parameter(horizon)  # Electricity price

    T_cond.value = df.loc[
        datetime:datetime+timedelta(hours=12), 'outdoor_temp'
    ].values

    e_price.value = df.loc[
        datetime:datetime+timedelta(hours=12), 'e_price'
    ].values

    # Initialize cost and constraints list
    cost = 0
    constraints = []
    constraints += [x[0] == 27.0]  # Initial storage energy

    for t in range(horizon):
        constraints += [rpm[t] <= 2900]  # Maximum rpm
        constraints += [rpm[t] >= 1200]  # Minimum rpm

        action = {
            'rpm': rpm[t],
            'T_cond': T_cond[t],
            'pcm_mode': u[t],
            'Q_discharge': Q_dis[t]
        }
        obs = system.step(action)
        EER = obs['EER']
        e = obs['energy_consumed']
        SoC = obs['storage_energy']  # update SoC

        constraints += [x[t + 1] == SoC]

        cost += e_price[t] * e


    # Define the constraints
    constraints = [x[0] == system.storage_energy]
    for t in range(horizon):
        constraints += [
            x[t + 1] == x[t] + u[t] * dt / 3600,
            0.0 <= u[t],
            u[t] <= 1.0
        ]

    # Define the objective function
    objective = cp.Minimize(cp.sum(u))

    # Create the optimization problem
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem
    problem.solve()

    # Extract the optimal action
    action = {
        'pcm_mode': u[0].value[0]
    }

    return action
