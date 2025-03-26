import pandas as pd
import os
from datetime import timedelta
from env.pcm_storage import hp_system
from models.mpc_15T import mpc_pso_15T


# load data
cwd = os.getcwd()
total_df = pd.read_pickle(os.path.join(cwd, 'data', 'total_df.pkl'))

start_date = pd.to_datetime('2021-06-01 00:00:00')
end_date = start_date + timedelta(days=31)

date = start_date

results = pd.DataFrame()

horizon = 6  # hours
dt = 900  # seconds

hp_system = hp_system(dt=dt)  # Initialize the HP system

rpm_init = 1200
soc_init = 0.0

while date < end_date - timedelta(hours=horizon):
    # Run the MPC controller
    res = mpc_pso_15T(
                    hp_system, horizon=horizon*4, dt=dt,
                    datetime_str=date, df=total_df,
                    rpm_init=rpm_init, soc_init=soc_init, Q_pcm=5.0,
                    iters=100, n_particles=50
    )

    print(f'Current date: {date}')
    date += timedelta(minutes=15)

    results = pd.concat(
        [results, res.iloc[0, :].to_frame().T],
        axis=0,
        ignore_index=True
    )

    rpm_init = res['rpm'].values[-1]
    soc_init = res['soc'].values[-1]

results.to_pickle('./results/results_15T.pkl')
