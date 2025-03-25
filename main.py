import pandas as pd
import os
import cvxpy as cp
from datetime import timedelta
from env.pcm_storage import hp_system
from models.mpc import mpc_controller, mpc_controller_pso


# load data
cwd = os.getcwd()
total_df = pd.read_pickle(os.path.join(cwd, 'data', 'total_df.pkl'))
total_df_hourly = pd.read_pickle(os.path.join(cwd, 'data', 'total_df_hourly.pkl'))

start_date = pd.to_datetime('2021-06-01 00:00:00')
end_date = start_date + timedelta(days=31)

date = start_date

results = pd.DataFrame()

hp_system = hp_system(dt=900)  # Initialize the HP system

rpm_init = 1500
soc_init = 0.0

while date < end_date - timedelta(hours=12):
    # Run the MPC controller
    res = mpc_controller_pso(
                            hp_system, horizon=12, dt=3600,
                            datetime_str=date, df=total_df_hourly,
                            rpm_init=rpm_init, soc_init=soc_init, Q_pcm=5.0,
                            iters=100, n_particles=50
    )

    print(f'Current date: {date}')
    date += timedelta(hours=1)

    results = pd.concat([results, res.iloc[0, :].to_frame().T], axis=0, ignore_index=True)

    rpm_init = res['rpm'].values[-1]
    soc_init = res['soc'].values[-1]

results.to_pickle('./results/results.pkl')
