import sys
import os
import numpy as np
import pandas as pd
import cvxpy as cp
from datetime import timedelta
import pyomo.environ as pyo

# This explicitly tells Python where to find your modules
sys.path.insert(0, os.path.abspath('.'))
print("Python Path:", sys.path[0])

# Import your module(s)
from env.pcm_storage import hp_system

# Initialize the HP system (using a time step of 900 seconds)
hp_system = hp_system(dt=3600)

# Load data (assumed to be a pickle file with a DataFrame)
df = pd.read_pickle('data/total_df_hourly.pkl')


def mpc_cvxpy_1H(hp_system, horizon=12, dt=3600,
                   datetime='2021-06-01 10:00:00', df=df,
                   soc_init=1.0, Q_pcm=5.0):
    # start_time = pd.to_datetime(datetime)

    # Decision variables
    rpm = cp.Variable(horizon, nonneg=True)
    u_hp = cp.Variable(horizon, boolean=True)
    u_rpm = cp.Variable(horizon, nonneg=True)

    soc = cp.Variable(horizon + 1, nonneg=True)
    u_pcm_charge = cp.Variable(horizon, boolean=True)
    u_pcm_discharge = cp.Variable(horizon, boolean=True)

    # Enforce mutual exclusivity for PCM: cannot charge and discharge at the same time
    constraints = []
    for t in range(horizon):
        constraints.append(u_pcm_charge[t] + u_pcm_discharge[t] <= 1)

    # Get outdoor temperature, electricity price, and load forecasts
    T_cond = df.loc[datetime:datetime + timedelta(hours=horizon - 1), 'outdoor_temp'].values
    e_price = df.loc[datetime:datetime + timedelta(hours=horizon - 1), 'e_price'].values * 0.001
    load = df.loc[datetime:datetime + timedelta(hours=horizon - 1), 'load'].values

    # Initial conditions
    constraints += [
        soc[0] == soc_init
    ]

    for t in range(horizon):
        constraints += [
            u_hp[t] >= u_rpm[t],  # rpm must be off if rpm is off
            rpm[t] == u_rpm[t] * (2900 - 1200) + 1200 * u_hp[t]  # rpm must be between 1200 and 2900 if rpm is on
            ]

    cost = 0
    penalty = 10  # penalty for skew rate

    for t in range(horizon):
        # PCM power contribution: positive for charging, negative for discharging.
        Q_dot_pcm = Q_pcm * (u_pcm_charge[t] - u_pcm_discharge[t])
        Q_action = Q_dot_pcm * dt / 3600.0

        # HP cooling power and efficiency calculations
        # First compute the cooling power expression
        Q_dot_cool = (hp_system.Q_intercept + hp_system.a * rpm[t] +
                      hp_system.b * T_cond[t] + hp_system.c * (rpm[t] ** 2) +
                      hp_system.d * (T_cond[t] ** 2))
        # Compute the EER expression (valid when the pump is on)
        EER_expr = (hp_system.EER_intercept + hp_system.e * rpm[t] +
                    hp_system.f * T_cond[t] + hp_system.g * (rpm[t] ** 2) +
                    hp_system.h * (T_cond[t] ** 2))
        # Force EER to be 0 when heat pump is off by multiplying with u_hp.
        # When u_hp[t]==0, Q_cool will be 0 so we define e_hp as 0.
        # To avoid division by zero, we use a small number in the denominator.
        Q_cool = Q_dot_cool * (dt / 3600.0) * u_hp[t]
        e_hp = cp.multiply(u_hp[t], Q_cool) / (EER_expr + 1e-6)

        # Update state of charge
        constraints += [
            soc[t + 1] == soc[t] + Q_action / 27.0
        ]
        # Energy balance: the sum of cooling power and PCM action must equal the load
        constraints += [
            Q_cool + Q_action == load[t]
        ]
        # SoC constraints
        constraints += [
            soc[t + 1] <= 1.0,
            soc[t + 1] >= 0.0
        ]
        cost += e_price[t] * e_hp

    for i in range(horizon-1):
        cost += penalty * (u_hp[i+1] - u_hp[i])**2

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.COPT)

    res = {
        'rpm': rpm.value,
        'Q_pcm': (Q_pcm * (u_pcm_charge.value - u_pcm_discharge.value)) * (dt / 3600.0),
        'soc': soc.value[:-1],
        'outdoor_temp': T_cond,
        'e_price': e_price,
        'load': load,
        'u_pcm_charge': u_pcm_charge.value,
        'u_pcm_discharge': u_pcm_discharge.value
    }

    soc_final = soc.value[-1]
    res_df = pd.DataFrame(res)

    return res_df, soc_final


def mpc_pyomo_1H(hp_system, horizon=12, dt=3600,
                 datetime='2021-06-01 10:00:00', df=None,
                 soc_init=1.0, Q_pcm=5.0):
    # Convert input datetime string and extract forecast data from df
    start_time = pd.to_datetime(datetime)
    df_subset = df.loc[start_time: start_time + timedelta(hours=horizon-1)]
    T_cond_array = df_subset['outdoor_temp'].values
    e_price_array = df_subset['e_price'].values * 0.001  # unit conversion as in CVXPY model
    load_array = df_subset['load'].values

    # Create the Pyomo model
    model = pyo.ConcreteModel()

    # Define sets: T for time steps [0, horizon-1] and T_soc for SoC [0, horizon]
    model.T = pyo.RangeSet(0, horizon-1)
    model.T_soc = pyo.RangeSet(0, horizon)

    # Parameters from forecast data (indexed by t)
    model.T_cond = pyo.Param(model.T, initialize=lambda m, t: T_cond_array[t])
    model.e_price = pyo.Param(model.T, initialize=lambda m, t: e_price_array[t])
    model.cooling_load = pyo.Param(model.T, initialize=lambda m, t: load_array[t])

    # Decision variables
    model.rpm = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.u_hp = pyo.Var(model.T, domain=pyo.Binary)
    # u_rpm is continuous and (by the constraint below) will be in [0,1]
    model.u_rpm = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.soc = pyo.Var(model.T_soc, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.u_pcm_charge = pyo.Var(model.T, domain=pyo.Binary)
    model.u_pcm_discharge = pyo.Var(model.T, domain=pyo.Binary)

    # Constraint: mutual exclusivity for PCM (cannot charge and discharge simultaneously)
    def pcm_mutual_exclusivity_rule(m, t):
        return m.u_pcm_charge[t] + m.u_pcm_discharge[t] <= 1
    model.pcm_mutual_exclusivity = pyo.Constraint(model.T, rule=pcm_mutual_exclusivity_rule)

    # Initial condition for state-of-charge
    model.soc[0].fix(soc_init)

    # Constraint: u_hp[t] >= u_rpm[t] (so if the heat pump is off, then the rpm control must be 0)
    def u_hp_ge_u_rpm_rule(m, t):
        return m.u_hp[t] >= m.u_rpm[t]
    model.u_hp_ge_u_rpm = pyo.Constraint(model.T, rule=u_hp_ge_u_rpm_rule)

    # Constraint: define rpm such that when the pump is on, rpm is between 1200 and 2900
    def rpm_definition_rule(m, t):
        return m.rpm[t] == m.u_rpm[t] * (2900 - 1200) + 1200 * m.u_hp[t]
    model.rpm_definition = pyo.Constraint(model.T, rule=rpm_definition_rule)

    # SoC update: soc[t+1] = soc[t] + (PCM energy action) / 27.0
    def soc_update_rule(m, t):
        Q_action = Q_pcm * (m.u_pcm_charge[t] - m.u_pcm_discharge[t]) * (dt / 3600.0)
        return m.soc[t+1] == m.soc[t] + Q_action / 27.0
    model.soc_update = pyo.Constraint(model.T, rule=soc_update_rule)

    # Energy balance: cooling provided by the heat pump plus PCM energy must meet the load.
    def energy_balance_rule(m, t):
        Q_action = Q_pcm * (m.u_pcm_charge[t] - m.u_pcm_discharge[t]) * (dt / 3600.0)
        Q_dot_cool = (hp_system.Q_intercept
                      + hp_system.a * m.rpm[t]
                      + hp_system.b * m.T_cond[t]
                      + hp_system.c * m.rpm[t]**2
                      + hp_system.d * m.T_cond[t]**2)
        Q_cool = Q_dot_cool * (dt / 3600.0) * m.u_hp[t]
        return Q_cool + Q_action == m.cooling_load[t]
    model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)

    # Define the objective function
    penalty = 10  # penalty factor for changes in the heat pump on/off status

    def objective_rule(m):
        total_cost = 0
        for t in m.T:
            # Compute cooling power from the heat pump
            Q_dot_cool = (hp_system.Q_intercept
                          + hp_system.a * m.rpm[t]
                          + hp_system.b * m.T_cond[t]
                          + hp_system.c * m.rpm[t]**2
                          + hp_system.d * m.T_cond[t]**2)
            Q_cool = Q_dot_cool * (dt / 3600.0) * m.u_hp[t]
            # Compute the EER (energy efficiency ratio) expression
            EER_expr = (hp_system.EER_intercept
                        + hp_system.e * m.rpm[t]
                        + hp_system.f * m.T_cond[t]
                        + hp_system.g * m.rpm[t]**2
                        + hp_system.h * m.T_cond[t]**2)
            # Avoid division by zero (as in CVXPY, a small number is added)
            e_hp = (m.u_hp[t] * Q_cool) / (EER_expr + 1e-6)
            total_cost += m.e_price[t] * e_hp
        # Add a penalty for changes in the heat pump's status
        for t in m.T:
            if t < horizon - 1:
                total_cost += penalty * (m.u_hp[t+1] - m.u_hp[t])**2
        return total_cost

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Solve the model.
    # (Note: Because this is an MINLP, you will need a solver like Couenne or Bonmin.)
    solver = pyo.SolverFactory('couenne')
    result = solver.solve(model, tee=True)

    # Extract the results into Python lists
    rpm_res = [pyo.value(model.rpm[t]) for t in model.T]
    Q_pcm_res = [Q_pcm * (pyo.value(model.u_pcm_charge[t]) - pyo.value(model.u_pcm_discharge[t])) * (dt / 3600.0)
                 for t in model.T]
    soc_res = [pyo.value(model.soc[t]) for t in sorted(model.T_soc)[:-1]]  # excluding the final soc value if desired
    u_pcm_charge_res = [pyo.value(model.u_pcm_charge[t]) for t in model.T]
    u_pcm_discharge_res = [pyo.value(model.u_pcm_discharge[t]) for t in model.T]

    # Compile results in a DataFrame
    res_dict = {
        'rpm': rpm_res,
        'Q_pcm': Q_pcm_res,
        'soc': soc_res,
        'outdoor_temp': T_cond_array,
        'e_price': e_price_array,
        'load': load_array,
        'u_pcm_charge': u_pcm_charge_res,
        'u_pcm_discharge': u_pcm_discharge_res
    }
    res_df = pd.DataFrame(res_dict)
    soc_final = pyo.value(model.soc[horizon])

    return res_df, soc_final
