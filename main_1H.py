import pandas as pd
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64
import os
from datetime import timedelta
from env.pcm_storage import hp_system
from models.mpc_1H import mpc_cvxpy_1H, mpc_pyomo_1H


# load data
data_dir = os.path.join(os.getcwd(), 'data')
total_df = pd.read_pickle(os.path.join(data_dir, 'total_df.pkl'))
total_df_hourly = pd.read_pickle(os.path.join(data_dir, 'total_df_hourly.pkl'))

start_date = pd.to_datetime('2021-06-01 00:00:00')
end_date = start_date + timedelta(days=10)

date = start_date

results = pd.DataFrame()

horizon = 6  # hours
dt = 3600  # seconds

hp_system = hp_system(dt=dt)  # Initialize the HP system

rpm_init = 1200
soc_init = 0.0

while date < end_date - timedelta(hours=horizon):
    # Run the MPC controller
    res, soc_init = mpc_pyomo_1H(
                    hp_system, horizon=horizon, dt=dt,
                    datetime=date, df=total_df_hourly,
                    soc_init=soc_init, Q_pcm=5.0
    )

    print(f'Current date: {date}')
    date += timedelta(hours=1)

    results = pd.concat(
        [results, res.iloc[0, :].to_frame().T],
        axis=0,
        ignore_index=True
    )

results.to_pickle('./results/results_1H.pkl')
