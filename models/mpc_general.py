import os
import pandas as pd
import numpy as np
from datetime import timedelta
from models.linear_hp import hp_invert  # hp_invert(Q, T) returns (rpm, EER, e_hp)


def mpc_cvxpy(horizon=48, dt=900, datetime=None, df=None, soc_init=0.0,
                  Q_dot_pcm=10.0, w_penalty=0.0):
    """
    MPC optimization for a 15-minute timestep horizon with a Pareto objective.

    The objective is the weighted sum of:
      1. The operational energy cost: sum(e_price[t] * e_hp[t])
      2. The average absolute change in HP rpm: (1/(horizon-1)) * sum(|rpm[t+1] - rpm[t]|)

    Parameters:
      horizon   : Number of time steps in the horizon.
      dt        : Time step in seconds (900 for 15 minutes).
      datetime  : Starting datetime as a string (default '2021-06-01 00:00:00').
      df        : DataFrame with forecast data (if None, a default file is loaded).
      soc_init  : Initial state-of-charge.
      Q_dot_pcm : PCM power limit (kW).
      w_penalty : Weight on the penalty term for rapid changes in HP rpm.
                  (Set weight_operational=1 and vary w_penalty to study trade-offs.)

    Returns:
      res_df           : DataFrame with columns:
                           - 'u_pcm'   : PCM power decision (kW)
                           - 'Q_action': PCM energy contribution (kWh)
                           - 'Q_cool'  : Required cooling power (kW)
                           - 'rpm'     : HP compressor speed (rpm)
                           - 'e_hp'    : HP energy consumption (kWh)
                           - 'soc'     : State-of-charge of TES (excluding final time step)
                           - 'e_price' : Electricity price at each time step
                           - 'load'    : Cooling load (kW)
                           - 'T_cond'  : Outdoor temperature (°C)
      soc_final        : Final state-of-charge after the horizon.
      energy_cost_val  : Total operational energy cost (sum over horizon).
      penalty_cost_val : Average rpm change over the horizon.
      overall_cost_val : Sum of energy cost + w_penalty * penalty_cost_val.
    """
    import cvxpy as cp
    # Set default datetime and load data if needed
    if datetime is None:
        datetime = '2021-06-01 00:00:00'
    if df is None:
        df = pd.read_pickle(os.path.join(os.getcwd(), 'data', 'total_df.pkl'))

    tes_capacity = 27.0 * 2.0  # Example TES capacity

    # Decision variable for PCM power (kW) over the horizon.
    u_pcm = cp.Variable(horizon)
    # State-of-charge (normalized) over the horizon.
    soc = cp.Variable(horizon + 1, nonneg=True)

    constraints = [soc[0] == soc_init]

    # Extract forecast data over the horizon
    start_datetime = pd.to_datetime(datetime)
    forecast_end = start_datetime + timedelta(seconds=dt*(horizon-1))
    T_cond = df.loc[datetime:forecast_end, 'outdoor_temp'].values
    e_price = df.loc[datetime:forecast_end, 'e_price'].values * 0.001  # adjust units if needed
    load = df.loc[datetime:forecast_end, 'load'].values

    # Add Gaussian noise to simulate forecast errors in outdoor temperature.
    T_cond_noisy = T_cond.copy()
    for i in range(horizon):
        factor = (i + 1) / horizon
        error_std_T = 1 * factor  # base std can be tuned
        error_T = np.random.normal(0, error_std_T)
        T_cond_noisy[i] += error_T
    T_cond = T_cond_noisy

    energy_cost_expr = 0
    penalty_expr = 0

    # Lists for building the output DataFrame.
    Q_action_expr_list = []
    Q_cool_expr_list = []
    rpm_expr_list = []
    e_hp_expr_list = []

    for t in range(horizon):
        # PCM power limits.
        constraints += [u_pcm[t] >= -Q_dot_pcm, u_pcm[t] <= Q_dot_pcm]

        # Convert PCM power (kW) to energy (kWh) over the dt period.
        Q_action = u_pcm[t] * (dt / 3600.0)
        Q_action_expr_list.append(Q_action)

        # Cooling power required is the sum of the load and the PCM action.
        Q_cool = load[t] + Q_action
        Q_cool_expr_list.append(Q_cool)

        # Compute HP characteristics using the inversion model.
        # hp_invert returns (rpm, EER, e_hp)
        rpm_val, EER_val, e_hp_val = hp_invert(Q_cool, T_cond[t])
        rpm_expr_list.append(rpm_val)
        e_hp_expr_list.append(e_hp_val)

        # Impose constraints: for instance, an upper bound on rpm and nonnegative Q_cool.
        constraints += [rpm_val <= 2900, Q_cool >= 0]

        # Update state-of-charge.
        constraints += [soc[t+1] == soc[t] + Q_action / tes_capacity,
                        soc[t+1] >= 0,
                        soc[t+1] <= 1.0]

        # Accumulate the operational energy cost.
        energy_cost_expr += e_price[t] * e_hp_val

    # Penalty term: average absolute change in hp rpm.
    for i in range(horizon - 1):
        penalty_expr += cp.abs(rpm_expr_list[i+1] - rpm_expr_list[i])
    avg_penalty_expr = penalty_expr / (horizon - 1)

    # Total cost is a weighted sum of the energy cost and the rpm change penalty.
    total_cost_expr = energy_cost_expr + w_penalty * avg_penalty_expr

    # Solve the optimization.
    problem = cp.Problem(cp.Minimize(total_cost_expr), constraints)
    problem.solve(solver=cp.COPT)

    # Extract numerical values.
    u_pcm_val = u_pcm.value
    soc_val = soc.value
    Q_action_vals = []
    Q_cool_vals = []
    rpm_vals = []
    e_hp_vals = []
    energy_cost_val = 0
    for t in range(horizon):
        Q_action_val = u_pcm_val[t] * (dt / 3600.0)
        Q_action_vals.append(Q_action_val)
        Q_cool_val = load[t] + Q_action_val
        Q_cool_vals.append(Q_cool_val)
        rpm_val, EER_val, e_hp_val = hp_invert(Q_cool_val, T_cond[t])
        rpm_vals.append(rpm_val)
        e_hp_vals.append(e_hp_val)
        energy_cost_val += e_price[t] * e_hp_val
    penalty_cost_val = np.mean(np.abs(np.diff(rpm_vals)))
    overall_cost_val = energy_cost_val + w_penalty * penalty_cost_val

    # Build results DataFrame.
    res = {
        'u_pcm': u_pcm_val,
        'Q_action': Q_action_vals,
        'Q_cool': Q_cool_vals,
        'rpm': rpm_vals,
        'e_hp': e_hp_vals,
        'soc': soc_val[:-1],
        'e_price': e_price,
        'load': load,
        'T_cond': T_cond
    }
    res_df = pd.DataFrame(res)
    soc_final = soc_val[1]

    return res_df, soc_final, energy_cost_val, penalty_cost_val, overall_cost_val
