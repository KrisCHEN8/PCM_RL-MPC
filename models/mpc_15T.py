import sys, os

# This explicitly tells Python where to find your modules
sys.path.insert(0, os.path.abspath('.'))

# Verify the path added
print("Python Path:", sys.path[0])

# Now import your module
import numpy as np
import pandas as pd
from datetime import timedelta
from env.pcm_storage import hp_system
import pyswarms as ps

hp_system = hp_system(dt=900)  # Initialize the HP system

df = pd.read_pickle('data/total_df.pkl')


def objective_function(x, hp_system, horizon, dt, start_time,
                       rpm_init, soc_init, Q_pcm, T_cond, e_price, load, day_mask):
    """
    Computes the cost for each particle. The decision vector ordering is:
      [rpm[0], ..., rpm[horizon], soc[0], ..., soc[horizon],
       u_pcm[0], ..., u_pcm[horizon-1], u_hp[0], ..., u_hp[horizon-1]]

    Constraints are imposed via penalty terms.
    """
    n_particles = x.shape[0]
    costs = np.zeros(n_particles)
    penalty_factor = 1e8
    hp_change_penalty = 5
    delta_rpm = 300

    for i in range(n_particles):
        particle = x[i, :]
        # Extract decision variables:
        # rpm: indices 0 to horizon (length horizon+1)
        rpm = particle[0:(horizon+1)]
        # soc: indices horizon+1 to 2*(horizon+1)-1
        soc = particle[(horizon+1):2*(horizon+1)]
        # u_pcm: next horizon values
        u_pcm_cont = particle[2*(horizon+1):2*(horizon+1)+horizon]
        # u_hp: final horizon values
        u_hp_cont = particle[2*(horizon+1)+horizon:]
        # Threshold the binary variables:
        u_pcm = (u_pcm_cont >= 0.5).astype(float)
        u_hp = (u_hp_cont >= 0.5).astype(float)

        cost = 0.0
        penalty = 0.0
        skew_rate = 0.0

        # Penalize deviations from the initial conditions
        penalty += penalty_factor * ((rpm[0] - rpm_init)**2)
        penalty += penalty_factor * ((soc[0] - soc_init)**2)

        # Enforce delta rpm constraint
        for t in range(horizon):
            diff = np.abs(rpm[t+1] - rpm[t])
            if diff > delta_rpm:
                penalty += penalty_factor * ((diff - delta_rpm)**2)

        # Loop over each time step
        for t in range(horizon):
            # Determine Q_dot_pcm and Q_action using the precomputed day mask:
            mask = day_mask[t]  # 1 if daytime (9-21), 0 otherwise
            Q_dot_pcm = Q_pcm * (1 - mask) - Q_pcm * mask
            Q_action = Q_dot_pcm * dt / 3600.0

            # HP cooling power and efficiency using rpm[t+1] and T_cond[t]:
            r = rpm[t+1]
            T = T_cond[t]
            Q_dot_cool = (hp_system.Q_intercept + hp_system.a * r +
                          hp_system.b * T + hp_system.c * (r**2) +
                          hp_system.d * (T**2))
            EER = (hp_system.EER_intercept + hp_system.e * r +
                   hp_system.f * T + hp_system.g * (r**2) +
                   hp_system.h * (T**2))
            # Avoid division by zero in EER:
            if np.abs(EER) < 1e-6:
                penalty += penalty_factor * 1e3
                EER = 1e-6
            Q_cool = Q_dot_cool * (dt / 3600.0) * u_hp[t]
            e_hp = Q_cool / EER

            # SoC update: soc[t+1] should equal soc[t] + u_pcm[t]*Q_action/27.0
            soc_pred = soc[t] + (u_pcm[t] * Q_action) / 27.0
            penalty += penalty_factor * ((soc[t+1] - soc_pred)**2)

            # Energy balance: Q_cool + u_pcm[t]*Q_action should equal load[t]
            penalty += penalty_factor * ((Q_cool + u_pcm[t]*Q_action - load[t])**2)

            # Enforce RPM limits at time t+1
            if r < 1200:
                penalty += penalty_factor * ((1200 - r)**2)
            if r > 2900:
                penalty += penalty_factor * ((r - 2900)**2)

            # Enforce SoC limits at time t+1
            if soc[t+1] < 0:
                penalty += penalty_factor * ((0 - soc[t+1])**2)
            if soc[t+1] > 1:
                penalty += penalty_factor * ((soc[t+1] - 1)**2)

            # Accumulate operating cost (energy price * HP energy consumption)
            cost += e_price[t] * e_hp

        # Control action skew rate penalty
        for t in range(horizon-1):
            skew_rate += hp_change_penalty * (u_hp[t+1] - u_hp[t])**2

        costs[i] = cost + penalty + skew_rate

    return costs

def mpc_pso_15T(hp_system, horizon=12, dt=900, datetime_str='2021-06-01 00:00:00',
                df=None, rpm_init=2000, soc_init=1.0, Q_pcm=5.0, iters=100,
                n_particles=50):
    """
    Solves the MPC problem using pyswarms.
    """
    if df is None:
        raise ValueError("A valid dataframe 'df' must be provided.")

    start_time = pd.to_datetime(datetime_str)
    # For 15-minute steps, compute the end_time accordingly:
    end_time = start_time + timedelta(seconds=dt*(horizon-1))
    # Extract the necessary data from the dataframe
    T_cond = df.loc[datetime_str:end_time, 'outdoor_temp'].values
    e_price = df.loc[datetime_str:end_time, 'e_price'].values * 0.001
    load = df.loc[datetime_str:end_time, 'load'].values

    # Precompute a binary mask for daytime (1 if between 9:00 and 21:00, else 0)
    day_mask = np.array([1 if 9 <= (start_time + timedelta(seconds=dt*t)).hour < 21 else 0
                         for t in range(horizon)])

    # Total decision vector dimension:
    # (horizon+1) for rpm + (horizon+1) for soc + horizon for u_pcm + horizon for u_hp
    dim = 2*(horizon+1) + 2*horizon  # equals 4*horizon + 2

    # Define lower and upper bounds for each decision variable:
    lb = np.zeros(dim)
    ub = np.zeros(dim)

    # rpm bounds: index 0 fixed to rpm_init; indices 1..horizon between 1200 and 2900.
    lb[0] = rpm_init
    ub[0] = rpm_init
    lb[1:(horizon+1)] = 1200
    ub[1:(horizon+1)] = 2900

    # soc bounds: index (horizon+1) fixed to soc_init; indices horizon+2..2*(horizon+1)-1 between 0 and 1.
    lb[horizon+1] = soc_init
    ub[horizon+1] = soc_init
    lb[horizon+2:2*(horizon+1)] = 0.0
    ub[horizon+2:2*(horizon+1)] = 1.0

    # u_pcm bounds (relaxed binary): length = horizon, between 0 and 1.
    lb[2*(horizon+1):2*(horizon+1)+horizon] = 0.0
    ub[2*(horizon+1):2*(horizon+1)+horizon] = 1.0

    # u_hp bounds (relaxed binary): length = horizon, between 0 and 1.
    lb[2*(horizon+1)+horizon:] = 0.0
    ub[2*(horizon+1)+horizon:] = 1.0

    bounds = (lb, ub)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    # Wrap the objective function for pyswarms:
    def obj_func(x):
        return objective_function(x, hp_system, horizon, dt, start_time,
                                  rpm_init, soc_init, Q_pcm, T_cond, e_price, load, day_mask)

    # Set up and run the PSO optimizer:
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dim,
                                          options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(obj_func, iters=iters)

    # Extract and parse the best solution:
    rpm_sol = best_pos[0:(horizon+1)]
    soc_sol = best_pos[(horizon+1):2*(horizon+1)]
    u_pcm_sol = best_pos[2*(horizon+1):2*(horizon+1)+horizon]
    u_hp_sol = best_pos[2*(horizon+1)+horizon:]
    # Threshold the binary variables
    u_pcm_sol = (u_pcm_sol >= 0.5).astype(float)
    u_hp_sol = (u_hp_sol >= 0.5).astype(float)

    # Compute Q_action for each time step
    Q_action_arr = np.zeros(horizon)
    for t in range(horizon):
        mask = day_mask[t]
        Q_dot_pcm = Q_pcm * (1 - mask) - Q_pcm * mask
        Q_action_arr[t] = Q_dot_pcm * dt / 3600.0

    res = {
        'rpm': rpm_sol[1:],
        'u_hp': u_hp_sol,
        'Q_pcm': Q_action_arr * u_pcm_sol,
        'soc': soc_sol[:-1],
        'outdoor_temp': T_cond,
        'e_price': e_price,
        'load': load,
        'u_pcm': u_pcm_sol
    }
    res_df = pd.DataFrame(res)

    return res_df
