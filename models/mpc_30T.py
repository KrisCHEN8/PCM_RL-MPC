import pandas as pd
import cvxpy as cp
from datetime import timedelta
from models.linear_hp import hp_invert

# Load data (assumed to be a pickle file with a DataFrame)
df = pd.read_pickle('data/total_df.pkl')


def mpc_cvxpy_30T(horizon=24, dt=1800, datetime=None, df=None, soc_init=0.0,
                  Q_dot_pcm=10.0):
    # Default datetime if not provided
    if datetime is None:
        datetime = '2021-06-01 00:00:00'

    if df is None:
        df = pd.read_pickle('data/total_df_30T.pkl')

    tes_capacity = 27.0 * 2.0

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

    cost = 0
    penalty = 10  # penalty for skew rate

    # Loop over the horizon to build constraints and cost
    for t in range(horizon):
        constraints += [
            u_pcm[t] >= -Q_dot_pcm,
            u_pcm[t] <= Q_dot_pcm
        ]

        # Conversion factor dt/3600 converts power to energy over the 30-min period
        Q_action = u_pcm[t] * (dt / 3600.0)

        # HP cooling power and efficiency calculations
        rpm, EER_expr, e_dot_cool = hp_invert(u_hp[t], T_cond[t])
        Q_cool = (dt / 3600.0) * u_hp[t]
        e_hp = e_dot_cool * (dt / 3600.0)

        # Update state of charge (SoC) and enforce energy balance
        constraints += [
            soc[t + 1] == soc[t] + Q_action / tes_capacity,
            Q_cool - Q_action == load[t],
            soc[t + 1] <= 1.0,
            soc[t + 1] >= 0.0
        ]
        cost += e_price[t] * e_hp

    # Add penalty for rapid changes in HP power
    for i in range(horizon - 1):
        cost += penalty * (u_hp[i+1] - u_hp[i])**2

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.COPT)

    res = {
        'u_hp': u_hp.value,
        'rpm': [hp_invert(u_hp.value[t], T_cond[t])[0] for t in range(horizon)],
        'Q_pcm': u_pcm.value * dt / 3600.0,
        'soc': soc.value[:-1],
        'outdoor_temp': T_cond,
        'e_price': e_price,
        'load': load,
        'Q_cool': u_hp.value * dt / 3600.0,
        'e_hp': [hp_invert(u_hp.value[t], T_cond[t])[2] for t in range(horizon)],
        'EER': [hp_invert(u_hp.value[t], T_cond[t])[1] for t in range(horizon)]
    }

    soc_final = soc.value[1]
    res_df = pd.DataFrame(res)

    return res_df, soc_final
