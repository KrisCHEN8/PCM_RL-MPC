import pandas as pd
import os
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import timedelta
from env.pcm_storage import hp_system, pcm_system
from models.mpc import mpc_controller, mpc_pso_controller


# load data
cwd = os.getcwd()
total_df = pd.read_pickle(os.path.join(cwd, 'data', 'total_df.pkl'))

start_date = pd.to_datetime('2021-08-01 09:00:00')
end_date = start_date + timedelta(days=31)

date = start_date

results = pd.DataFrame()

hp_system = hp_system(dt=900)  # Initialize the HP system

while date < end_date:
    pcm = pcm_system(dt=900, SoC=27.0)  # Initialize the PCM storage

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
    res = mpc_pso_controller(hp_system, horizon=12, df=total_df)

    date += timedelta(hours=24)
    print(res)

    results = pd.concat([results, res], axis=0, ignore_index=True)
