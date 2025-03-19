import numpy as np
import pandas as pd
import os
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import timedelta
from env.pcm_storage import PCMStorage


system = PCMStorage(dt=900, initial_storage=32.0)


def mpc_controller(system, horizon=4, dt=900, datetime='2022-08-31', df=None):
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
    load = cp.Parameter(horizon)  # Building load

    T_cond.value = df.loc[
        datetime:datetime+timedelta(hours=12), 'outdoor_temp'
    ].values

    e_price.value = df.loc[
        datetime:datetime+timedelta(hours=12), 'e_price'
    ].values

    load.value = df.loc[
        datetime:datetime+timedelta(hours=12), 'load'
    ].values

    # Initialize cost and constraints list
    cost = 0
    constraints = []
    constraints += [x[0] == 27.0]  # Initial storage energy

    EER = []
    e = []
    SoC = []

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
        EER += obs['EER']
        e += obs['energy_consumed']
        SoC += obs['storage_energy']  # update SoC

        constraints += [x[t + 1] == SoC]

        constraints += [x[t + 1] <= 27.0]  # SoC limits

        constraints += [Q_dis[t] <= 5]  # Maximum discharge power

        cost += e_price[t] * e[-1]

    objective = cp.Minimize(cost)
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem
    problem.solve()

    # Extract the optimal states and actions
    res = {
        'pcm_mode': u.value,
        'rpm': rpm.value,
        'Q_discharge': Q_dis.value,
        'EER': EER,
        'energy_consumed': e,
        'storage_energy': x.value[:-1],
        'outdoor_temp': T_cond.value,
        'e_price': e_price.value
    }

    system.reset(initial_storage=32.0)

    return res
