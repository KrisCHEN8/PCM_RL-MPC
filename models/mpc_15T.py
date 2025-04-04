import pandas as pd
import cvxpy as cp
from datetime import timedelta
from models.linear_hp import hp_invert


def mpc_cvxpy_15T(horizon=48, dt=900, datetime=None, df=None, soc_init=0.0,
                  Q_dot_pcm=10.0, w_penalty=10.0, rpm_changing_rate=None):
    """
    MPC optimization using cvxpy with a variable penalty weight (w_penalty).

    Parameters:
      horizon     : number of time steps in the horizon
      dt          : time step in seconds (e.g., 900 for 15 minutes)
      datetime    : starting datetime as string (default '2021-06-01 00:00:00')
      df          : DataFrame containing forecast data (if None, a default file is loaded)
      soc_init    : initial state of charge
      Q_dot_pcm   : PCM power limit (both directions)
      w_penalty   : penalty weight for rapid changes in HP power

    Returns:
      res_df          : DataFrame with optimization results (first-step outputs, etc.)
      soc_final       : state of charge at the end of the first step
      energy_cost_val : total energy cost computed from the solution
      penalty_cost_val: total penalty cost computed from the solution
      overall_cost_val: overall cost (energy cost + w_penalty * penalty cost)
    """
    # Default datetime if not provided
    if datetime is None:
        datetime = '2021-06-01 00:00:00'
    # Load the data if not provided
    if df is None:
        # Use a data file that has 15-minute resolution (update file name as needed)
        df = pd.read_pickle('data/total_df_15min.pkl')

    tes_capacity = 27.0 * 2.0  # example TES capacity

    # Decision variables
    u_hp = cp.Variable(horizon, nonneg=True)  # HP power
    u_pcm = cp.Variable(horizon)              # PCM power
    soc = cp.Variable(horizon + 1, nonneg=True)

    constraints = []

    # Adjust the forecast extraction window to match the dt in seconds
    forecast_end = pd.to_datetime(datetime) + timedelta(seconds=dt*(horizon - 1))
    T_cond = df.loc[datetime:forecast_end, 'outdoor_temp'].values
    e_price = df.loc[datetime:forecast_end, 'e_price'].values * 0.001
    load = df.loc[datetime:forecast_end, 'load'].values

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
        Q_cool = (dt / 3600.0) * u_hp[t]
        # Cooling energy over the period (Wh)
        e_hp = e_dot_cool * (dt / 3600.0)

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
    for i in range(horizon - 1):
        penalty_cost_val += (u_hp.value[i+1] - u_hp.value[i])**2

    overall_cost_val = energy_cost_val + w_penalty * penalty_cost_val

    # Build result dictionary (you can expand this as needed)
    res = {
        'u_hp': u_hp.value,
        'rpm': rpm_list,
        'Q_pcm': u_pcm.value * dt / 3600.0,
        'soc': soc.value[:-1],
        'outdoor_temp': T_cond,
        'e_price': e_price,
        'load': load,
        'Q_cool': u_hp.value * dt / 3600.0,
        'e_hp': [hp_invert(u_hp.value[t], T_cond[t])[2] * (dt/3600.0) for t in range(horizon)],
        'EER': [hp_invert(u_hp.value[t], T_cond[t])[1] for t in range(horizon)]
    }

    soc_final = soc.value[1]
    res_df = pd.DataFrame(res)

    return res_df, soc_final, energy_cost_val, penalty_cost_val, overall_cost_val
