import pandas as pd
import cvxpy as cp
import numpy as np
from datetime import timedelta
from models.linear_hp import hp_invert

def mpc_cvxpy(horizon=48, dt=900, datetime=None, df=None, soc_init=0.0,
              Q_dot_pcm=10.0, w_penalty=10.0, rpm_changing_rate=None):
    """
    MPC optimization using cvxpy with a variable penalty weight (w_penalty).

    Parameters:
      horizon     : Number of time steps in the horizon.
      dt          : Time step in seconds (e.g., 900 for 15 minutes, 1800 for 30 minutes, or 3600 for 1 hour).
      datetime    : Starting datetime as a string (default '2021-06-01 00:00:00').
      df          : DataFrame containing forecast data (if None, a default file is loaded).
      soc_init    : Initial state of charge.
      Q_dot_pcm   : PCM power limit (both charging and discharging).
      w_penalty   : Penalty weight for rapid changes in HP power.

    Returns:
      res_df          : DataFrame with optimization results (first-step outputs, etc.).
      soc_final       : State of charge at the end of the first step.
      energy_cost_val : Total energy cost computed from the solution.
      penalty_cost_val: Total penalty cost computed from the solution.
      overall_cost_val: Overall cost (energy cost + w_penalty * penalty cost).
    """
    # Default datetime if not provided
    if datetime is None:
        datetime = '2021-06-01 00:00:00'
    # Load the data if not provided (ensure that the data has the correct resolution)
    if df is None:
        df = pd.read_pickle('data/total_df.pkl')

    tes_capacity = 27.0 * 2.0  # Example TES capacity

    # Decision variables
    u_hp = cp.Variable(horizon, nonneg=True)  # HP power
    u_pcm = cp.Variable(horizon)              # PCM power
    soc = cp.Variable(horizon + 1, nonneg=True)

    constraints = []

    # Adjust the forecast extraction window to match the dt in seconds
    start_datetime = pd.to_datetime(datetime)
    forecast_end = start_datetime + timedelta(seconds=dt * (horizon - 1))
    T_cond = df.loc[datetime:forecast_end, 'outdoor_temp'].values
    e_price = df.loc[datetime:forecast_end, 'e_price'].values * 0.001
    load = df.loc[datetime:forecast_end, 'load'].values

    # Create a list of timestamps for each forecast step
    time_stamps = [start_datetime + timedelta(seconds=i*dt) for i in range(horizon)]

    # Add Gaussian noise to simulate forecast errors.
    T_cond_noisy = T_cond.copy()
    load_noisy = load.copy()
    for i in range(horizon):
        factor = (i + 1) / horizon  # Scale factor increases with time step

        # Add noise to T_cond at every time step.
        error_std_T = 2 * factor   # Base standard deviation for temperature error is 2°C (scaled)
        error_T = np.random.normal(0, error_std_T)
        T_cond_noisy[i] += error_T

        # Check if the current time step is on a working day (Mon-Fri) and during working hours (9am to 7pm)
        current_time = time_stamps[i]
        if current_time.weekday() < 5 and 9 <= current_time.hour < 19:
            error_std_load = 1 * factor  # Base standard deviation for load error is 3 kWh (scaled)
            error_load = np.random.normal(0, error_std_load)
            load_noisy[i] += error_load

    # Use the noisy forecasts for the optimization
    T_cond = T_cond_noisy
    load = load_noisy

    # Initial condition
    constraints += [soc[0] == soc_init]

    # Build cost expression (separated into energy and penalty components)
    energy_cost_expr = 0
    rpm_list = []
    for t in range(horizon):
        # PCM power constraints
        constraints += [
            u_pcm[t] >= -Q_dot_pcm,
            u_pcm[t] <= Q_dot_pcm
        ]

        # Convert PCM power (kW) into energy (kWh) over dt
        Q_action = u_pcm[t] * (dt / 3600.0)

        # HP model: compute rpm, EER, and cooling energy rate (e_dot_cool)
        rpm, EER_expr, e_dot_cool = hp_invert(u_hp[t], T_cond[t])
        Q_cool = u_hp[t] * (dt / 3600.0)  # Cooling energy (kWh) over dt
        e_hp = e_dot_cool * (dt / 3600.0)   # Cooling energy over the period (Wh)

        # State-of-charge update and energy balance constraints
        constraints += [
            soc[t + 1] == soc[t] + Q_action / tes_capacity,
            Q_cool - Q_action == load[t],
            soc[t + 1] <= 1.0,
            soc[t + 1] >= 0.0
        ]
        rpm_list.append(rpm)
        energy_cost_expr += e_price[t] * e_hp

    # Compute the penalty term for rapid changes in HP power
    penalty_cost_expr = 0
    for i in range(horizon - 1):
        penalty_cost_expr += cp.abs(rpm_list[i+1] - rpm_list[i])

    # Total cost: energy cost + penalty weight * penalty cost
    total_cost = energy_cost_expr + w_penalty * penalty_cost_expr

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(total_cost), constraints)
    problem.solve(solver=cp.MOSEK)

    # Calculate numerical cost components using the solved values
    energy_cost_val = 0
    for t in range(horizon):
        _, _, e_dot_cool_val = hp_invert(u_hp.value[t], T_cond[t])
        e_hp_val = e_dot_cool_val * (dt / 3600.0)
        energy_cost_val += e_price[t] * e_hp_val

    penalty_cost_val = 0

    # Build result dictionary (you can expand this as needed)
    res = {
        'u_hp': u_hp.value,
        'rpm': [hp_invert(u_hp.value[i], T_cond[i])[0] for i in range(horizon)],
        # 'u_pcm': u_pcm.value,
        # 'soc': soc.value[:-1],
        'outdoor_temp': T_cond
        # 'e_price': e_price,
        # 'load': load,
        # 'Q_cool': u_hp.value * dt / 3600.0,
        # 'e_hp': [hp_invert(u_hp.value[t], T_cond[t])[2] * (dt / 3600.0) for t in range(horizon)],
        # 'EER': [hp_invert(u_hp.value[t], T_cond[t])[1] for t in range(horizon)]
    }

    res_df = pd.DataFrame(res)

    return res_df, energy_cost_val, penalty_cost_val
