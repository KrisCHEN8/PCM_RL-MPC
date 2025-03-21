import pandas as pd
import os
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import timedelta
from env.pcm_storage import PCMHeatPumpSystem
from models.mpc import mpc_controller


# load data
cwd = os.getcwd()
total_df = pd.read_pickle(os.path.join(cwd, 'data', 'total_df.pkl'))

start_date = pd.to_datetime('2021-06-01 09:00:00')
end_date = start_date + timedelta(days=31)

date = start_date

system = PCMHeatPumpSystem(dt=900, initial_storage=27.0)

results = pd.DataFrame()

while date < end_date:
    # Run the MPC controller
    res = mpc_controller(system,
                         horizon=12*4,
                         dt=900,
                         datetime=date,
                         df=total_df)
    date += timedelta(hours=24)
    print(res)

    results = pd.concat([results, res], axis=0, ignore_index=True)
