import pandas as pd
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64
import os
from datetime import timedelta
from models.mpc_general import mpc_cvxpy
# from env.hp_env import hp_env, pcm_env
from models.linear_hp import hp_invert


# load data
data_dir = os.path.join(os.getcwd(), 'data')
total_df = pd.read_pickle(os.path.join(data_dir, 'total_df_30T.pkl'))

start_date = pd.to_datetime('2021-08-01 00:00:00')
end_date = start_date + timedelta(days=31)

date = start_date

results = pd.DataFrame()

dt = 1800  # seconds
horizon = int(12 * (3600 / dt))

soc_init = 0.0

while date < end_date:
    # Run the MPC controller
    res = mpc_cvxpy(
        horizon=horizon,
        dt=dt,
        datetime=date,
        df=total_df,
        soc_init=soc_init,
        Q_dot_pcm=10.0,
        w_penalty=0.001
    )[0]

    # rpm_hp = res['rpm'][0]
    T_cond = total_df.loc[date, 'outdoor_temp']
    # u_hp = res['u_hp'][0]
    u_pcm = res['u_pcm'][0]
    load = res['load'][0]
    e_price = res['e_price'][0]

    # Update the state of the HP
    e_hp = hp_invert(load + u_pcm * (dt / 3600), T_cond)[2]
    EER = hp_invert(load + u_pcm * (dt / 3600), T_cond)[1]
    cost = e_price * e_hp
    tes_capacity = 27.0 * 2.0  # Example TES capacity
    # Q_cool = Q_dot_cool * dt / 3600

    # # Update the state of the PCM
    # Q_pcm = u_pcm
    # Q_dot_pcm = Q_pcm * 3600 / dt

    # # Update the state of the PCM
    # soc_init = pcm_env(Q_dot_pcm, dt, soc_init)

    print(f'Current date: {date}')
    date += timedelta(seconds=dt)

    res_real = {
        'u_pcm': u_pcm,
        'Q_cool': load + u_pcm * (dt / 3600),
        'soc': soc_init,
        'outdoor_temp': T_cond,
        'load': load,
        'e_price': e_price,
        'e_hp': e_hp,
        'cost': cost,
        'EER': EER
    }

    soc_init = soc_init + ((u_pcm * (dt / 3600)) / tes_capacity)

    real_df = pd.DataFrame(res_real, index=[date])

    results = pd.concat(
        [results, real_df],
        axis=0,
        ignore_index=True
    )

results.to_pickle('./results/results_30T.pkl')
