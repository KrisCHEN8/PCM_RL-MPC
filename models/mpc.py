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
        hp_system, pcm_system,
        horizon=4*12, dt=900,
        datetime='2021-06-01 00:00:00', df=df
):
    rpm = cp.Variable(horizon, nonneg=True)
    Q_dot_disc = cp.Variable(horizon, nonneg=True)
    soc = cp.Variable(horizon + 1)

    T_cond = df.loc[
        datetime:datetime+timedelta(hours=horizon/4)-timedelta(minutes=15),
        'outdoor_temp'].values

    e_price = df.loc[
        datetime:datetime+timedelta(hours=horizon/4)-timedelta(minutes=15),
        'e_price'].values

    load = df.loc[
        datetime:datetime+timedelta(hours=horizon/4)-timedelta(minutes=15),
        'load'].values

    constraints = [soc[0] == 27.0]
    cost = 0

    for t in range(horizon):
        # HP Cooling power (symbolically)
        Q_dot_cool = (hp_system.Q_intercept + hp_system.a * rpm[t] +
                      hp_system.b * T_cond[t] + hp_system.c * rpm[t]**2 +
                      hp_system.d * T_cond[t]**2)

        EER = (hp_system.EER_intercept + hp_system.e * rpm[t] +
               hp_system.f * T_cond[t] + hp_system.g * rpm[t]**2 +
               hp_system.h * T_cond[t]**2)

        Q_cool = Q_dot_cool * dt / 3600.0
        e_hp = Q_cool / EER

        # PCM storage update
        constraints += [soc[t+1] == soc[t] - Q_dot_disc[t] * dt / 3600.0]

        # Energy balance constraint
        constraints += [Q_dot_cool + Q_dot_disc[t] == load[t]]

        # RPM limits
        constraints += [rpm[t] <= 2900, rpm[t] >= 1200]

        # PCM constraints
        constraints += [
            soc[t+1] <= 27.0,
            soc[t+1] >= 0.0,
            Q_dot_disc[t] <= 5.0]

        # Accumulate cost
        cost += e_price[t] * e_hp

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    res = {
        'rpm': rpm.value,
        'Q_discharge': Q_dot_disc.value,
        'storage_energy': soc.value[:-1],
        'outdoor_temp': T_cond,
        'e_price': e_price
    }

    return res


def mpc_pso_controller(hp_system, horizon=12, dt=900,
                       datetime='2021-06-01 00:00:00', df=None):

    import numpy as np
    import pandas as pd
    import pyswarms as ps
    from datetime import timedelta

    def fitness(x, hp_system, horizon, dt, T_cond, e_price, load, initial_soc=27.0):
        n_particles = x.shape[0]
        rpm = x[:, :horizon]
        Q_dot_disc = x[:, horizon:]

        penalty = np.zeros(n_particles)
        cost = np.zeros(n_particles)

        for i in range(n_particles):
            soc = np.zeros(horizon + 1)
            soc[0] = initial_soc

            for t in range(horizon):
                Q_dot_cool = (
                    hp_system.Q_intercept + hp_system.a * rpm[i, t] +
                    hp_system.b * T_cond[t] + hp_system.c * rpm[i, t] ** 2 +
                    hp_system.d * T_cond[t] ** 2
                )

                EER = (
                    hp_system.EER_intercept + hp_system.e * rpm[i, t] +
                    hp_system.f * T_cond[t] + hp_system.g * rpm[i, t] ** 2 +
                    hp_system.h * T_cond[t] ** 2
                )

                Q_cool = Q_dot_cool * dt / 3600.0
                e_hp = Q_cool / EER

                soc[t + 1] = soc[t] - Q_dot_disc[i, t] * dt / 3600.0

                penalty[i] += 1e4 * np.abs(max(0, rpm[i, t] - 2900))
                penalty[i] += 1e4 * np.abs(max(0, 1200 - rpm[i, t]))
                penalty[i] += 1e4 * np.abs(max(0, Q_dot_disc[i, t] - 5))
                penalty[i] += 1e4 * (max(0, soc[t + 1] - 27.0) + max(0, -soc[t + 1]))

                energy_balance = Q_dot_cool + Q_dot_disc[i, t] - load[t]
                penalty[i] += 1e4 * np.abs(energy_balance)

                cost[i] += e_price[t] * e_hp

        total_cost = cost + penalty

        return total_cost

    datetime_obj = pd.to_datetime(datetime)

    T_cond = df.loc[
        datetime_obj:datetime_obj+timedelta(hours=horizon/4)-timedelta(minutes=15),
        'outdoor_temp'].values

    e_price = df.loc[
        datetime_obj:datetime_obj+timedelta(hours=horizon/4)-timedelta(minutes=15),
        'e_price'].values

    load = df.loc[
        datetime_obj:datetime_obj+timedelta(hours=horizon/4)-timedelta(minutes=15),
        'load'].values

    lb = np.concatenate((np.full(horizon, 1200), np.zeros(horizon)))
    ub = np.concatenate((np.full(horizon, 2900), np.full(horizon, 5)))

    bounds = (lb, ub)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=50, dimensions=2 * horizon,
        options={'c1': 1.5, 'c2': 1.5, 'w': 0.5}, bounds=bounds
    )

    best_cost, best_pos = optimizer.optimize(
        fitness, iters=100, hp_system=hp_system, horizon=horizon, dt=dt,
        T_cond=T_cond, e_price=e_price, load=load
    )

    rpm_optimal = best_pos[:horizon]
    Q_discharge_optimal = best_pos[horizon:]

    soc = np.zeros(horizon + 1)
    soc[0] = 27.0
    for t in range(horizon):
        soc[t + 1] = soc[t] - Q_discharge_optimal[t] * dt / 3600.0

    res = pd.DataFrame({
        'rpm': rpm_optimal,
        'Q_discharge': Q_discharge_optimal,
        'storage_energy': soc[:-1],
        'outdoor_temp': T_cond,
        'e_price': e_price,
        'best_cost': np.full(horizon, best_cost)
    })

    return res
