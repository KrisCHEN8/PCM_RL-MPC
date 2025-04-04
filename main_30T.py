import pandas as pd
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64
import os
from datetime import timedelta
from models.mpc_general import mpc_cvxpy


# load data
data_dir = os.path.join(os.getcwd(), 'data')
total_df = pd.read_pickle(os.path.join(data_dir, 'total_df_30T.pkl'))

start_date = pd.to_datetime('2021-07-01 00:00:00')
end_date = start_date + timedelta(days=31)

date = start_date

results = pd.DataFrame()

dt = 1800  # seconds
horizon = int(12 * (dt / 3600))

soc_init = 0.0

while date < end_date - timedelta(hours=horizon):
    # Run the MPC controller
    res, soc_init = mpc_cvxpy(
        horizon=horizon,
        dt=dt,
        datetime=date,
        df=total_df,
        soc_init=soc_init,
        Q_dot_pcm=10.0,
        w_penalty=0.1,
        rpm_changing_rate=300
    )[0:2]

    print(f'Current date: {date}')
    date += timedelta(minutes=30)

    results = pd.concat(
        [results, res.iloc[0, :].to_frame().T],
        axis=0,
        ignore_index=True
    )

results.to_pickle('./results/results_30T.pkl')
