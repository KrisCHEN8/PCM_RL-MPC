import pandas as pd
import os
import cvxpy as cp
from datetime import timedelta
from env.pcm_storage import hp_system
from models.mpc import mpc_controller, mpc_pso_controller


# load data
cwd = os.getcwd()
total_df = pd.read_pickle(os.path.join(cwd, 'data', 'total_df.pkl'))
total_df_hourly = pd.read_pickle(os.path.join(cwd, 'data', 'total_df_hourly.pkl'))

start_date = pd.to_datetime('2021-05-01 09:00:00')
end_date = start_date + timedelta(days=31)

date = start_date

results = pd.DataFrame()

hp_system = hp_system(dt=900)  # Initialize the HP system

rpm_init = 2000
soc_init = 0.0

while date < end_date - timedelta(hours=12):
    # Run the MPC controller
    '''
    res = mpc_controller(
        hp_system,
        pcm_system,
        horizon=12*4,
        dt=900,
        datetime=date,
        df=total_df
        )
    '''
    res = mpc_pso_controller(hp_system, horizon=12,
                             dt=900, datetime=date, df=total_df_hourly,
                             rpm_init=rpm_init, soc_init=soc_init)

    date += timedelta(hours=1)
    print(res)

    results = pd.concat([results, res.iloc[0, :]], axis=0, ignore_index=True)

    rpm_init = res['rpm'].values[-1]
    soc_init = res['soc'].values[-1]

results.to_pickle('./results/results.pkl')
