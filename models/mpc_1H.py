import sys, os
import numpy as np
import pandas as pd
import cvxpy as cp
from datetime import timedelta
import pyswarms as ps
import pyomo.environ as pyo

# This explicitly tells Python where to find your modules
sys.path.insert(0, os.path.abspath('.'))
print("Python Path:", sys.path[0])

# Import your module(s)
from env.pcm_storage import hp_system, pcm_system

# Initialize the HP system (using a time step of 900 seconds)
hp_system = hp_system(dt=900)

# Load data (assumed to be a pickle file with a DataFrame)
df = pd.read_pickle('data/total_df_hourly.pkl')


def mpc_controller(hp_system, horizon=12, dt=3600,
                   datetime='2021-06-01 10:00:00', df=df,
                   rpm_init=2000, soc_init=1.0, Q_pcm=5.0):
    start_time = pd.to_datetime(datetime)

    # Decision variables
    rpm = cp.Variable(horizon + 1, nonneg=True)
    soc = cp.Variable(horizon + 1, nonneg=True)
    # Binary variables for PCM charging/discharging and heat pump status
    u_pcm_charge = cp.Variable(horizon, boolean=True)
    u_pcm_discharge = cp.Variable(horizon, boolean=True)
    u_hp = cp.Variable(horizon, boolean=True)

    # Enforce mutual exclusivity for PCM: cannot charge and discharge at the same time
    constraints = []
    for t in range(horizon):
        constraints.append(u_pcm_charge[t] + u_pcm_discharge[t] <= 1)

    # Get outdoor temperature, electricity price, and load forecasts
    T_cond = df.loc[datetime:datetime + timedelta(hours=horizon - 1), 'outdoor_temp'].values
    e_price = df.loc[datetime:datetime + timedelta(hours=horizon - 1), 'e_price'].values * 0.001
    load = df.loc[datetime:datetime + timedelta(hours=horizon - 1), 'load'].values

    # Initial conditions
    constraints += [
        rpm[0] == rpm_init,
        soc[0] == soc_init
    ]

    delta_rpm = 500  # maximum allowed change between time steps
    for t in range(horizon):
        constraints += [cp.abs(rpm[t + 1] - rpm[t]) <= delta_rpm]
        # If heat pump is off (u_hp[t]==0), then force rpm[t+1]==0.
        # If on (u_hp[t]==1), then rpm[t+1] must be between 1200 and 2900.
        constraints += [
            rpm[t + 1] <= 2900 * u_hp[t],
            rpm[t + 1] >= 1200 * u_hp[t]
        ]

    cost = 0
    for t in range(horizon):
        time_step = start_time + timedelta(hours=t)
        # Define a simple daytime mask (if needed)
        day_start, day_end = 9, 21
        daytime_mask = 1 if day_start <= time_step.hour < day_end else 0

        # PCM power contribution: positive for charging, negative for discharging.
        Q_dot_pcm = Q_pcm * u_pcm_charge[t] - Q_pcm * u_pcm_discharge[t]
        Q_action = Q_dot_pcm * dt / 3600.0

        # HP cooling power and efficiency calculations
        # First compute the cooling power expression
        Q_dot_cool = (hp_system.Q_intercept + hp_system.a * rpm[t + 1] +
                      hp_system.b * T_cond[t] + hp_system.c * (rpm[t + 1] ** 2) +
                      hp_system.d * (T_cond[t] ** 2))
        # Compute the EER expression (valid when the pump is on)
        EER_expr = (hp_system.EER_intercept + hp_system.e * rpm[t + 1] +
                    hp_system.f * T_cond[t] + hp_system.g * (rpm[t + 1] ** 2) +
                    hp_system.h * (T_cond[t] ** 2))
        # Force EER to be 0 when heat pump is off by multiplying with u_hp.
        # When u_hp[t]==0, Q_cool will be 0 so we define e_hp as 0.
        # To avoid division by zero, we use a small number in the denominator.
        Q_cool = Q_dot_cool * (dt / 3600.0) * u_hp[t]
        e_hp = cp.multiply(u_hp[t], Q_cool) / (EER_expr + 1e-6)

        # Update state of charge
        constraints += [
            soc[t + 1] == soc[t] + Q_action / 27.0
        ]
        # Energy balance: the sum of cooling power and PCM action must equal the load
        constraints += [
            Q_cool + Q_action == load[t]
        ]
        # SoC constraints
        constraints += [
            soc[t + 1] <= 1.0,
            soc[t + 1] >= 0.0
        ]
        cost += e_price[t] * e_hp

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    res = {
        'rpm': rpm.value[1:],
        'u_hp': u_hp.value,
        'Q_pcm': (Q_pcm * (u_pcm_charge.value - u_pcm_discharge.value)) * (dt / 3600.0),
        'soc': soc.value[:-1],
        'outdoor_temp': T_cond,
        'e_price': e_price,
        'load': load,
        'u_pcm_charge': u_pcm_charge.value,
        'u_pcm_discharge': u_pcm_discharge.value
    }

    res_df = pd.DataFrame(res)
    return res_df


def objective_function(x, hp_system, horizon, dt, start_time,
                       rpm_init, soc_init, Q_pcm, T_cond, e_price, load):
    """
    Computes the cost for each particle. The decision vector ordering is:
      [rpm[0], ..., rpm[horizon],
       soc[0], ..., soc[horizon],
       u_pcm_charge[0], ..., u_pcm_charge[horizon-1],
       u_pcm_discharge[0], ..., u_pcm_discharge[horizon-1],
       u_hp[0], ..., u_hp[horizon-1]]
    Constraints are imposed via penalty terms.
    """
    n_particles = x.shape[0]
    costs = np.zeros(n_particles)
    penalty_factor = 1e18
    delta_rpm = 1000

    for i in range(n_particles):
        particle = x[i, :]
        # Extract decision variables
        rpm = particle[0:(horizon + 1)]
        soc = particle[(horizon + 1):2 * (horizon + 1)]
        start_charge = 2 * (horizon + 1)
        end_charge = start_charge + horizon
        u_pcm_charge_cont = particle[start_charge:end_charge]
        start_discharge = end_charge
        end_discharge = start_discharge + horizon
        u_pcm_discharge_cont = particle[start_discharge:end_discharge]
        u_hp_cont = particle[end_discharge:]
        # Threshold the relaxed binary variables
        u_pcm_charge = (u_pcm_charge_cont >= 0.5).astype(float)
        u_pcm_discharge = (u_pcm_discharge_cont >= 0.5).astype(float)
        u_hp = (u_hp_cont >= 0.5).astype(float)

        cost = 0.0
        penalty = 0.0

        # Initial condition penalties
        penalty += penalty_factor * ((rpm[0] - rpm_init) ** 2)
        penalty += penalty_factor * ((soc[0] - soc_init) ** 2)

        # Enforce exclusivity for PCM modes: u_pcm_charge + u_pcm_discharge <= 1
        for t in range(horizon):
            if u_pcm_charge[t] + u_pcm_discharge[t] > 1:
                penalty += penalty_factor * ((u_pcm_charge[t] + u_pcm_discharge[t] - 1) ** 2)

        for t in range(horizon):
            Q_dot_pcm = Q_pcm * u_pcm_charge[t] - Q_pcm * u_pcm_discharge[t]
            Q_action = Q_dot_pcm * (dt / 3600.0)
            r = rpm[t + 1]
            T = T_cond[t]
            Q_dot_cool = (hp_system.Q_intercept + hp_system.a * r +
                          hp_system.b * T + hp_system.c * (r ** 2) +
                          hp_system.d * (T ** 2))
            Q_cool = Q_dot_cool * (dt / 3600.0) * u_hp[t]

            # Compute the EER based on whether the heat pump is on or off.
            EER_expr = (hp_system.EER_intercept + hp_system.e * r +
                        hp_system.f * T + hp_system.g * (r ** 2) +
                        hp_system.h * (T ** 2))
            if u_hp[t] < 0.5:
                # When the heat pump is off, force EER and e_hp to be 0
                EER_val = 0
                e_hp = 0
            else:
                EER_val = EER_expr
                # Avoid division by zero if EER_val is extremely small
                if abs(EER_val) < 1e-6:
                    penalty += penalty_factor * 1e3
                    EER_val = 1e-6
                e_hp = Q_cool / EER_val

            # SoC update consistency
            soc_pred = soc[t] + Q_action / 27.0
            penalty += penalty_factor * ((soc[t + 1] - soc_pred) ** 2)

            # Energy balance: HP cooling plus PCM action must equal load
            penalty += penalty_factor * ((Q_cool + Q_action - load[t]) ** 2)

            # Enforce rpm constraints: if the heat pump is off then r must be 0,
            # otherwise r must be within [1200, 2900]
            if u_hp[t] < 0.5:
                penalty += penalty_factor * (r ** 2)
            else:
                if r < 1200:
                    penalty += penalty_factor * ((1200 - r) ** 2)
                if r > 2900:
                    penalty += penalty_factor * ((r - 2900) ** 2)

            cost += e_price[t] * e_hp

        costs[i] = cost + penalty

    return costs


def mpc_pso_1H(hp_system, horizon=12, dt=3600, datetime_str='2021-06-01 10:00:00',
               df=None, rpm_init=2000, soc_init=1.0, Q_pcm=5.0, iters=100,
               n_particles=50):
    """
    Solves the MPC problem using pyswarms with the modified PCM control
    """
    if df is None:
        raise ValueError("A valid dataframe 'df' must be provided.")

    start_time = pd.to_datetime(datetime_str)
    end_time = start_time + timedelta(hours=horizon - 1)
    T_cond = df.loc[datetime_str:end_time, 'outdoor_temp'].values
    e_price = df.loc[datetime_str:end_time, 'e_price'].values * 0.001
    load = df.loc[datetime_str:end_time, 'load'].values

    # Decision vector dimensions:
    # rpm: horizon+1, soc: horizon+1, u_pcm_charge: horizon, u_pcm_discharge: horizon, u_hp: horizon
    dim = 5 * horizon + 2

    lb = np.zeros(dim)
    ub = np.zeros(dim)

    # RPM bounds: first index fixed; subsequent rpm values (when heat pump is on) between 0 and 2900.
    lb[0] = rpm_init
    ub[0] = rpm_init
    lb[1:(horizon + 1)] = 0
    ub[1:(horizon + 1)] = 2900

    # SoC bounds: first soc fixed; remaining between 0 and 1.
    lb[horizon + 1] = soc_init
    ub[horizon + 1] = soc_init
    lb[horizon + 2:2 * (horizon + 1)] = 0.0
    ub[horizon + 2:2 * (horizon + 1)] = 1.0

    # u_pcm_charge bounds (relaxed binary)
    start_charge = 2 * (horizon + 1)
    lb[start_charge:start_charge + horizon] = 0.0
    ub[start_charge:start_charge + horizon] = 1.0

    # u_pcm_discharge bounds (relaxed binary)
    start_discharge = start_charge + horizon
    lb[start_discharge:start_discharge + horizon] = 0.0
    ub[start_discharge:start_discharge + horizon] = 1.0

    # u_hp bounds (relaxed binary)
    start_u_hp = start_discharge + horizon
    lb[start_u_hp:start_u_hp + horizon] = 0.0
    ub[start_u_hp:start_u_hp + horizon] = 1.0

    bounds = (lb, ub)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    def obj_func(x):
        return objective_function(x, hp_system, horizon, dt, start_time,
                                  rpm_init, soc_init, Q_pcm, T_cond, e_price, load)

    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dim,
                                          options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(obj_func, iters=iters)

    rpm_sol = best_pos[0:(horizon + 1)]
    soc_sol = best_pos[(horizon + 1):2 * (horizon + 1)]
    u_pcm_charge_sol = best_pos[2 * (horizon + 1):2 * (horizon + 1) + horizon]
    u_pcm_discharge_sol = best_pos[2 * (horizon + 1) + horizon:2 * (horizon + 1) + 2 * horizon]
    u_hp_sol = best_pos[2 * (horizon + 1) + 2 * horizon:]
    u_pcm_charge_sol = (u_pcm_charge_sol >= 0.5).astype(float)
    u_pcm_discharge_sol = (u_pcm_discharge_sol >= 0.5).astype(float)
    u_hp_sol = (u_hp_sol >= 0.5).astype(float)

    Q_action_arr = np.zeros(horizon)
    for t in range(horizon):
        Q_dot_pcm = Q_pcm * u_pcm_charge_sol[t] - Q_pcm * u_pcm_discharge_sol[t]
        Q_action_arr[t] = Q_dot_pcm * dt / 3600.0

    res = {
        'rpm': rpm_sol[1:],
        'u_hp': u_hp_sol,
        'Q_pcm': Q_action_arr,
        'soc': soc_sol[:-1],
        'outdoor_temp': T_cond,
        'e_price': e_price,
        'load': load
    }
    res_df = pd.DataFrame(res)

    return res_df


def mpc_pyomo_1H(hp_system, horizon=12, dt=3600, datetime_str='2021-06-01 10:00:00',
                 df=None, rpm_init=2000, soc_init=1.0, Q_pcm=5.0, solver_name='bonmin'):
    """
    Solves the MPC problem using Pyomo and returns a DataFrame with the results.
    
    Parameters:
      hp_system : object or dict with attributes: Q_intercept, a, b, c, d,
                  EER_intercept, e, f, g, h.
      horizon   : Number of control intervals (default 12).
      dt        : Time step in seconds (default 3600).
      datetime_str: Start time as a string.
      df        : A pandas DataFrame with columns 'outdoor_temp', 'e_price', and 'load'.
      rpm_init  : Initial rpm value.
      soc_init  : Initial state-of-charge.
      Q_pcm     : Fixed PCM power rate.
      solver_name: MINLP solver to use (e.g., 'bonmin' or 'couenne').
    
    Returns:
      res_df    : A pandas DataFrame containing the solution values.
                The DataFrame includes:
                  - 'rpm': the rpm values for t=1,...,horizon,
                  - 'soc': the state-of-charge for t=0,...,horizon-1,
                  - 'u_hp': binary heat pump on/off decisions for each interval,
                  - 'Q_pcm': the PCM energy action for each interval,
                  - 'outdoor_temp', 'e_price', 'load' from the input dataframe.
    """
    # Check that the dataframe is provided.
    if df is None:
        raise ValueError("A valid dataframe 'df' must be provided.")

    # Extract input data from the dataframe for the control horizon.
    start_time = pd.to_datetime(datetime_str)
    end_time = start_time + timedelta(hours=horizon - 1)
    T_cond_arr = df.loc[datetime_str:end_time, 'outdoor_temp'].values
    e_price_arr = df.loc[datetime_str:end_time, 'e_price'].values * 0.001
    load_arr = df.loc[datetime_str:end_time, 'load'].values

    # Ensure the arrays are of length "horizon"
    if not (len(T_cond_arr) == horizon and len(e_price_arr) == horizon and len(load_arr) == horizon):
        raise ValueError("Dataframe does not have the required data length for the given horizon.")

    # Create a Pyomo model.
    model = pyo.ConcreteModel()

    # Define sets:
    #   T: time steps for state variables, from 0 to horizon.
    #   T_control: control intervals from 0 to horizon-1.
    model.T = pyo.RangeSet(0, horizon)
    model.T_control = pyo.RangeSet(0, horizon - 1)

    # Parameters:
    model.dt = pyo.Param(initialize=dt)
    model.Q_pcm = pyo.Param(initialize=Q_pcm)
    model.rpm_init = pyo.Param(initialize=rpm_init)
    model.soc_init = pyo.Param(initialize=soc_init)

    # Data parameters for each control interval:
    model.T_cond = pyo.Param(model.T_control, initialize=lambda m, t: float(T_cond_arr[t]))
    model.e_price = pyo.Param(model.T_control, initialize=lambda m, t: float(e_price_arr[t]))
    # Rename the 'load' parameter to 'load_demand' to avoid conflicts.
    model.load_demand = pyo.Param(model.T_control, initialize=lambda m, t: float(load_arr[t]))

    # HP system parameters:
    model.hp_Q_intercept = pyo.Param(initialize=hp_system.Q_intercept)
    model.hp_a = pyo.Param(initialize=hp_system.a)
    model.hp_b = pyo.Param(initialize=hp_system.b)
    model.hp_c = pyo.Param(initialize=hp_system.c)
    model.hp_d = pyo.Param(initialize=hp_system.d)
    model.hp_EER_intercept = pyo.Param(initialize=hp_system.EER_intercept)
    model.hp_e = pyo.Param(initialize=hp_system.e)
    model.hp_f = pyo.Param(initialize=hp_system.f)
    model.hp_g = pyo.Param(initialize=hp_system.g)
    model.hp_h = pyo.Param(initialize=hp_system.h)

    # Decision Variables:
    #   rpm[t] for t in 0,...,horizon (with rpm[0] fixed to rpm_init).
    model.rpm = pyo.Var(model.T, bounds=(0, 2900))
    #   soc[t] for t in 0,...,horizon.
    model.soc = pyo.Var(model.T, bounds=(0, 1))
    #   Binary variables for PCM charging/discharging and heat pump operation.
    model.u_pcm_charge = pyo.Var(model.T_control, domain=pyo.Binary)
    model.u_pcm_discharge = pyo.Var(model.T_control, domain=pyo.Binary)
    model.u_hp = pyo.Var(model.T_control, domain=pyo.Binary)
    #   Auxiliary variable for heat pump energy consumption.
    model.e_hp = pyo.Var(model.T_control, domain=pyo.NonNegativeReals)

    # Initial conditions.
    def init_rpm_rule(m):
        return m.rpm[0] == m.rpm_init
    model.rpm_init_con = pyo.Constraint(rule=init_rpm_rule)

    def init_soc_rule(m):
        return m.soc[0] == m.soc_init
    model.soc_init_con = pyo.Constraint(rule=init_soc_rule)

    # SoC update: soc[t+1] = soc[t] + (PCM action)/27, where PCM action = Q_pcm*(u_pcm_charge - u_pcm_discharge)*(dt/3600)
    def soc_update_rule(m, t):
        return m.soc[t + 1] == m.soc[t] + (m.Q_pcm * (m.u_pcm_charge[t] - m.u_pcm_discharge[t]) * (m.dt / 3600.0)) / 27.0
    model.soc_update_con = pyo.Constraint(model.T_control, rule=soc_update_rule)

    # PCM Exclusivity: cannot charge and discharge at the same time.
    def pcm_exclusivity_rule(m, t):
        return m.u_pcm_charge[t] + m.u_pcm_discharge[t] <= 1
    model.pcm_exclusivity_con = pyo.Constraint(model.T_control, rule=pcm_exclusivity_rule)

    # Energy balance: PCM contribution + HP cooling must equal the load.
    def energy_balance_rule(m, t):
        Q_action = m.Q_pcm * (m.u_pcm_charge[t] - m.u_pcm_discharge[t]) * (m.dt / 3600.0)
        Q_dot_cool = (m.hp_Q_intercept +
                      m.hp_a * m.rpm[t + 1] +
                      m.hp_b * m.T_cond[t] +
                      m.hp_c * (m.rpm[t + 1] ** 2) +
                      m.hp_d * (m.T_cond[t] ** 2))
        Q_cool = Q_dot_cool * (m.dt / 3600.0) * m.u_hp[t]
        return Q_action + Q_cool == m.load_demand[t]
    model.energy_balance_con = pyo.Constraint(model.T_control, rule=energy_balance_rule)

    # RPM constraints: when the heat pump is on, rpm must be within [1200, 2900]; otherwise, it is forced to 0.
    def rpm_lower_bound_rule(m, t):
        return m.rpm[t + 1] >= 1200 * m.u_hp[t]
    model.rpm_lower_bound_con = pyo.Constraint(model.T_control, rule=rpm_lower_bound_rule)

    def rpm_upper_bound_rule(m, t):
        return m.rpm[t + 1] <= 2900 * m.u_hp[t]
    model.rpm_upper_bound_con = pyo.Constraint(model.T_control, rule=rpm_upper_bound_rule)

    # Heat pump energy consumption: e_hp * EER_expr = Q_cool.
    def e_hp_rule(m, t):
        Q_dot_cool = (m.hp_Q_intercept +
                      m.hp_a * m.rpm[t + 1] +
                      m.hp_b * m.T_cond[t] +
                      m.hp_c * (m.rpm[t + 1] ** 2) +
                      m.hp_d * (m.T_cond[t] ** 2))
        Q_cool = Q_dot_cool * (m.dt / 3600.0) * m.u_hp[t]
        EER_expr = (m.hp_EER_intercept +
                    m.hp_e * m.rpm[t + 1] +
                    m.hp_f * m.T_cond[t] +
                    m.hp_g * (m.rpm[t + 1] ** 2) +
                    m.hp_h * (m.T_cond[t] ** 2))
        return m.e_hp[t] * EER_expr == Q_cool
    model.e_hp_con = pyo.Constraint(model.T_control, rule=e_hp_rule)

    # -------------------------------------------------
    # Objective Function: Minimize the total energy cost.
    def objective_rule(m):
        return sum(m.e_price[t] * m.e_hp[t] for t in m.T_control)
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # -------------------------------------------------
    # Solve the model
    # -------------------------------------------------
    solver = pyo.SolverFactory(solver_name)
    results = solver.solve(model, tee=False)

    # -------------------------------------------------
    # Extract results
    # -------------------------------------------------
    rpm_sol = [pyo.value(model.rpm[t]) for t in model.T]         # t = 0,...,horizon (t=0 is initial)
    soc_sol = [pyo.value(model.soc[t]) for t in model.T]           # t = 0,...,horizon
    u_hp_sol = [pyo.value(model.u_hp[t]) for t in model.T_control]  # t = 0,...,horizon-1
    u_pcm_charge_sol = [pyo.value(model.u_pcm_charge[t]) for t in model.T_control]
    u_pcm_discharge_sol = [pyo.value(model.u_pcm_discharge[t]) for t in model.T_control]
    e_hp_sol = [pyo.value(model.e_hp[t]) for t in model.T_control]

    # Compute the PCM action (energy) for each control interval:
    Q_action_arr = []
    for t in range(horizon):
        Q_action = Q_pcm * (u_pcm_charge_sol[t] - u_pcm_discharge_sol[t]) * (dt / 3600.0)
        Q_action_arr.append(Q_action)

    # Create the results dictionary.
    # Note: we discard the initial rpm (t=0) and the final soc (t=horizon) to align with the control intervals.
    res = {
        'rpm': rpm_sol[1:],            # rpm for t = 1,...,horizon
        'soc': soc_sol[:-1],           # soc for t = 0,...,horizon-1
        'u_hp': u_hp_sol,              # heat pump on/off decisions per control interval
        'Q_pcm': Q_action_arr,         # PCM energy action per interval
        'outdoor_temp': list(T_cond_arr),
        'e_price': list(e_price_arr),
        'load': list(load_arr)
    }
    res_df = pd.DataFrame(res)

    return res_df
