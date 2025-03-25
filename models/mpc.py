import numpy as np
import pandas as pd
import cvxpy as cp
from datetime import timedelta
from env.pcm_storage import hp_system, pcm_system
import pyswarms as ps


hp_system = hp_system(dt=900)  # Initialize the HP system
pcm_system = pcm_system(dt=900, SoC=27.0)  # Initialize the PCM storage

df = pd.read_pickle('data/total_df.pkl')


def mpc_controller(
        hp_system,
        horizon=6, dt=900,
        datetime='2021-06-01 00:00:00', df=df,
        rpm_init=2000, soc_init=27.0
):
    rpm = cp.Variable(horizon + 1, nonneg=True)
    Q_dot_disc = cp.Variable(horizon, nonneg=True)
    Q_dot_char = cp.Variable(horizon, nonneg=True)
    soc = cp.Variable(horizon + 1)
    # u_disc = cp.Variable(horizon, boolean=True)
    u_charge = cp.Variable(horizon, boolean=True)

    T_cond = df.loc[
        datetime:datetime+timedelta(hours=horizon-1),
        'outdoor_temp'].values

    e_price = df.loc[
        datetime:datetime+timedelta(hours=horizon-1),
        'e_price'].values

    load = df.loc[
        datetime:datetime+timedelta(hours=horizon-1),
        'load'].values

    constraints = [soc[0] == soc_init,
                   rpm[0] == rpm_init]

    cost = 0

    delta_rpm = 500  # maximum allowed change between time steps
    for t in range(horizon):
        constraints += [cp.abs(rpm[t+1] - rpm[t]) <= delta_rpm]

    for t in range(horizon):
        # HP Cooling power
        Q_dot_cool = (hp_system.Q_intercept + hp_system.a * rpm[t+1] +
                      hp_system.b * T_cond[t] + hp_system.c * rpm[t+1]**2 +
                      hp_system.d * T_cond[t]**2)

        EER = (hp_system.EER_intercept + hp_system.e * rpm[t+1] +
               hp_system.f * T_cond[t] + hp_system.g * rpm[t+1]**2 +
               hp_system.h * T_cond[t]**2)

        Q_cool = Q_dot_cool * dt / 3600.0
        e_hp = Q_cool / EER  # Electricity consumption in kWh

        # PCM storage update
        constraints += [
            soc[t+1] == soc[t] - (Q_dot_disc[t] * dt / 3600.0) / (27.0) +
            (Q_dot_char[t] * dt / 3600.0) / (32.0)
        ]

        bigM = 1e6
    # Energy balance constraints enforcing exclusive modes:
        constraints += [
            # When u_charge[t]==0 (supply load mode):
            Q_dot_cool + Q_dot_disc[t] >= load[t] - bigM * u_charge[t],
            Q_dot_cool + Q_dot_disc[t] <= load[t] + bigM * u_charge[t],

            # When u_charge[t]==1 (charging mode), load must be zero:
            load[t] <= bigM * (1 - u_charge[t]),

            # Enforce HP cooling used for charging in charging mode:
            Q_dot_char[t] <= Q_dot_cool + bigM*(1 - u_charge[t]),
            Q_dot_char[t] >= Q_dot_cool - bigM*(1 - u_charge[t]),
            ]

    # Limits for discharge and charge according to modes:
        constraints += [
            Q_dot_disc[t] <= 5.0 * (1 - u_charge[t]),
            Q_dot_char[t] <= 5.0 * u_charge[t]
            ]

        # RPM limits
        constraints += [rpm[t+1] <= 2900, rpm[t+1] >= 1200]

        # PCM constraints
        constraints += [
            soc[t+1] <= 1.0,
            soc[t+1] >= 0.0,
            Q_dot_disc[t] <= 5.0]

        # Accumulate cost
        cost += e_price[t] * e_hp

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    res = {
        'rpm': rpm.value[1:],
        'Q_discharge': Q_dot_disc.value,
        'storage_energy': soc.value[:-1],
        'outdoor_temp': T_cond,
        'e_price': e_price
    }

    res_df = pd.DataFrame(res)

    return res_df


import numpy as np
import pandas as pd
import pyswarms as ps
from datetime import timedelta


def mpc_pso_controller(
    hp_system,
    horizon=6, dt=900,
    datetime='2021-06-01 00:00:00', df=None,
    rpm_init=2000, soc_init=27.0
):

    T_cond = df.loc[
        datetime:datetime+timedelta(hours=horizon-1),
        'outdoor_temp'].values

    e_price = df.loc[
        datetime:datetime+timedelta(hours=horizon-1),
        'e_price'].values

    load = df.loc[
        datetime:datetime+timedelta(hours=horizon-1),
        'load'].values

    n_dim = 4 * horizon  # rpm, Q_dot_disc, Q_dot_char, u_charge

    lb = np.concatenate([
        np.full(horizon, 1200),        # rpm lower bound
        np.zeros(horizon),             # Q_dot_disc lower bound
        np.zeros(horizon),             # Q_dot_char lower bound
        np.zeros(horizon)              # u_charge lower bound (binary)
    ])

    ub = np.concatenate([
        np.full(horizon, 2900),        # rpm upper bound
        np.full(horizon, 5.0),         # Q_dot_disc upper bound
        np.full(horizon, 5.0),         # Q_dot_char upper bound
        np.ones(horizon)               # u_charge upper bound (binary)
    ])

    def fitness(x):
        n_particles = x.shape[0]
        penalty = np.zeros(n_particles)
        cost = np.zeros(n_particles)

        for i in range(n_particles):
            rpm = np.concatenate(([rpm_init], x[i, :horizon]))
            Q_dot_disc = x[i, horizon:2*horizon]
            Q_dot_char = x[i, 2*horizon:3*horizon]
            u_charge = np.round(x[i, 3*horizon:]).astype(int)

            soc = np.zeros(horizon + 1)
            soc[0] = soc_init

            for t in range(horizon):
                Q_dot_cool = (hp_system.Q_intercept + hp_system.a * rpm[t+1] +
                              hp_system.b * T_cond[t] + hp_system.c * rpm[t+1]**2 +
                              hp_system.d * T_cond[t]**2)

                EER = (hp_system.EER_intercept + hp_system.e * rpm[t+1] +
                       hp_system.f * T_cond[t] + hp_system.g * rpm[t+1]**2 +
                       hp_system.h * T_cond[t]**2)

                Q_cool = Q_dot_cool * dt / 3600.0
                e_hp = Q_cool / EER

                soc[t+1] = soc[t] - (Q_dot_disc[t] * dt / 3600.0)/27.0 + (Q_dot_char[t] * dt / 3600.0)/32.0

                # Constraints (penalty terms)
                penalty[i] += 1e6 * max(0, abs(rpm[t+1] - rpm[t]) - 500)
                penalty[i] += 1e6 * max(0, soc[t+1]-1.0) + 1e6 * max(0, -soc[t+1])

                # Energy balance constraints
                if u_charge[t] == 0:
                    penalty[i] += 1e6 * abs(Q_dot_cool + Q_dot_disc[t] - load[t])
                    penalty[i] += 1e6 * Q_dot_char[t]  # charging off
                else:
                    penalty[i] += 1e6 * abs(load[t])
                    penalty[i] += 1e6 * abs(Q_dot_char[t] - Q_dot_cool)
                    penalty[i] += 1e6 * Q_dot_disc[t]  # discharging off

                cost[i] += e_price[t] * e_hp

        return cost + penalty

    # Run optimization
    optimizer = ps.single.GlobalBestPSO(
        n_particles=50, dimensions=n_dim,
        options={'c1': 1.5, 'c2': 1.5, 'w': 0.5},
        bounds=(lb, ub)
    )

    best_cost, best_pos = optimizer.optimize(fitness, iters=200)

    rpm_opt = best_pos[:horizon]
    Q_disc_opt = best_pos[horizon:2*horizon]
    Q_char_opt = best_pos[2*horizon:3*horizon]
    u_charge_opt = np.round(best_pos[3*horizon:]).astype(int)

    soc_opt = np.zeros(horizon + 1)
    soc_opt[0] = soc_init
    for t in range(horizon):
        soc_opt[t+1] = soc_opt[t] - (Q_disc_opt[t] * dt / 3600.0)/27.0 + (Q_char_opt[t] * dt / 3600.0)/32.0

    res_df = pd.DataFrame({
        'rpm': rpm_opt,
        'Q_discharge': Q_disc_opt,
        'Q_charge': Q_char_opt,
        'u_charge': u_charge_opt,
        'storage_energy': soc_opt[:-1],
        'outdoor_temp': T_cond,
        'e_price': e_price,
        'best_cost': best_cost
    })

    return res_df

