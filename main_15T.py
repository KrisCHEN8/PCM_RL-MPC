import pandas as pd
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64
import os
from datetime import timedelta
from models.mpc_general import mpc_cvxpy
from env.hp_env import hp_env, pcm_env


# load data
data_dir = os.path.join(os.getcwd(), 'data')
total_df = pd.read_pickle(os.path.join(data_dir, 'total_df.pkl'))

start_date = pd.to_datetime('2021-07-01 00:00:00')
end_date = start_date + timedelta(days=31)

date = start_date

results = pd.DataFrame()

dt = 900  # seconds
horizon = int(12 * (dt / 3600))

soc_init = 0.0

while date < end_date - timedelta(hours=horizon):
    # Run the MPC controller
    res = mpc_cvxpy(
        horizon=horizon,
        dt=dt,
        datetime=date,
        df=total_df,
        soc_init=soc_init,
        Q_dot_pcm=10.0,
        w_penalty=0.1,
        rpm_changing_rate=300
    )[0]

    rpm_hp = res['rpm'][0]
    T_cond = res['outdoor_temp'][0]

    # Update the state of the HP
    Q_dot_cool, EER, e_hp = hp_env(rpm_hp, T_cond)
    Q_cool = Q_dot_cool * dt / 3600

    # Update the state of the PCM
    Q_pcm = max(total_df.loc[date, 'load'] - Q_cool, 0)
    Q_dot_pcm = Q_pcm * 3600 / dt

    # Update the state of the PCM
    soc_init = pcm_env(Q_dot_pcm, dt, soc_init)

    print(f'Current date: {date}')
    date += timedelta(minutes=15)

    res_real = {
        'u_hp': Q_dot_cool,
        'rpm': rpm_hp,
        'u_pcm': Q_dot_pcm,
        'soc': soc_init,
        'outdoor_temp': T_cond,
        'load': total_df.loc[date, 'load'],
        'Q_cool': Q_cool,
        'e_hp': e_hp,
        'EER': EER
    }

    real_df = pd.DataFrame(res_real, index=[date])

    results = pd.concat(
        [results, real_df],
        axis=0,
        ignore_index=True
    )

results.to_pickle('./results/results_15T.pkl')
