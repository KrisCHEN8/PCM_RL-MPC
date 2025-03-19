class PCMHeatPumpSystem:
    def __init__(self, dt=900.0, initial_storage=32.0):
        """
        Initialize the PCM thermal storage-HP system.

        Parameters:
            dt (float): Time step in seconds.
            initial_storage (float): Initial stored energy in kWh.
        """
        self.dt = dt  # time step in seconds
        self.storage_energy = initial_storage  # PCM storage energy in kWh

        # Timer to keep track of how long the system has been in charging mode
        self.charging_time = 0.0  # in seconds

        # PCM storage parameters (in kWh)
        self.full_charge_energy = 32.0  # Total energy required for full charge
        self.max_discharge_energy = 27.0  # Maximum retrievable energy during discharge

        # Coefficients for the heat pump (HP) cooling capacity correlation (Q_cool in kW)
        self.Q_intercept = -0.2071864002858792
        self.a = 0.005333811621408454      # coefficient for rpm
        self.b = -0.1365379343864041       # coefficient for T_cond
        self.c = -7.001133469771883e-07     # coefficient for rpm^2
        self.d = 0.0012415247870363416      # coefficient for T_cond^2

        # Coefficients for the HP EER correlation (unitless)
        self.EER_intercept = 4.421297840489518
        self.e = 0.009091070826281583       # coefficient for rpm
        self.f = -0.38184371581722676      # coefficient for T_cond
        self.g = -1.8923608336165108e-06    # coefficient for rpm^2
        self.h = 0.003420273670974565       # coefficient for T_cond^2

    def compute_Q_cool(self, rpm, T_cond):
        """
        Compute the cooling capacity (in kW) using the polynomial correlation.
        """
        return (self.Q_intercept +
                self.a * rpm +
                self.b * T_cond +
                self.c * (rpm ** 2) +
                self.d * (T_cond ** 2))

    def compute_EER(self, rpm, T_cond):
        """
        Compute the Energy Efficiency Ratio (EER) using the polynomial
        correlation.
        """
        return (self.EER_intercept +
                self.e * rpm +
                self.f * T_cond +
                self.g * (rpm ** 2) +
                self.h * (T_cond ** 2))

    def step(self, action):
        """
        Advance the system by one time step.

        Parameters:
            action (dict): Dictionary of control inputs. Expected keys:
                - 'rpm' (float): Heat pump rpm.
                - 'T_cond' (float): Condenser temperature.
                - 'pcm_mode' (float): PCM mode selector.
                     >0.1: Charging mode,
                     <-0.1: Discharging mode,
                     between -0.1 and 0.1: Normal operation.
                - 'Q_discharge' (float, optional): Desired discharge cooling
                  power (kW) when discharging.

        Returns:
            observation (dict): Contains updated state and outputs:
                - 'storage_energy' (kWh): Current stored energy.
                - 'charging_time' (s): Current charging duration.
                - 'Q_cool' (kW): Cooling power delivered in this step.
                - 'energy_consumed' (kWh): Energy consumed by the heat pump
                  in this step.
        """
        # Use a threshold to determine mode based on pcm_mode input
        pcm_mode_input = action.get('pcm_mode', 0)
        if pcm_mode_input == 0:
            mode = 'normal'
        elif pcm_mode_input == 1:
            mode = 'discharging'
        else:
            mode = 'charging'

        # Initialize outputs
        Q_cool = 0.0              # Cooling power delivered in kW
        energy_consumed = 0.0     # HP energy consumption in kWh
        hp_energy_output = 0.0    # HP energy output (kWh) in this time step
        EER = 0.0  # default initialization

        if mode == 'normal':
            # Normal HP operation without using PCM storage
            rpm = action.get('rpm', 0)
            T_cond = action.get('T_cond', 0)
            Q_cool = self.compute_Q_cool(rpm, T_cond)
            EER = self.compute_EER(rpm, T_cond)

            print(Q_cool, EER)

            # Cooling energy delivered during the time step (kWh)
            cooling_energy = Q_cool * self.dt / 3600.0
            # Electricity consumed by the heat pump
            energy_consumed = cooling_energy / EER if EER != 0 else 0.0

            # In normal mode, PCM storage state is unchanged;
            # also, reset any ongoing charging timer.
            self.charging_time = 0.0

        elif mode == 'charging':
            # Heat pump operates to charge the PCM storage.
            rpm = action.get('rpm', 0)
            T_cond = action.get('T_cond', 0)
            self.charging_time += self.dt

            # For the first 15000 s use the polynomial correlation;
            # afterward use constant 4 kW.
            if self.charging_time <= 15000:
                Q_hp = self.compute_Q_cool(rpm, T_cond)
                EER = self.compute_EER(rpm, T_cond)
            else:
                Q_hp = 4.0  # constant cooling power (kW)
                # For simplicity, we continue to use the computed EER.
                EER = self.compute_EER(rpm, T_cond)

            # Energy output by the heat pump in this step (kWh)
            hp_energy_output = Q_hp * self.dt / 3600.0
            energy_consumed = hp_energy_output / EER if EER != 0 else 0.0

            # Charge the PCM storage but do not exceed the full-charge
            # limit.
            energy_to_charge = min(hp_energy_output, self.full_charge_energy - self.storage_energy)
            self.storage_energy += energy_to_charge

            # Report the HP cooling power as the current output
            Q_cool = Q_hp

        elif mode == 'discharging':
            # PCM storage discharges to provide cooling.
            # The desired cooling power (kW) should be provided in the action.
            Q_discharge_requested = action.get('Q_discharge', 0)

            # Determine how much energy is available to discharge.
            # Even if the storage is full (or more than 27 kWh), only 27 kWh is retrievable.
            self.storage_energy = min(self.storage_energy, self.max_discharge_energy)

            # Calculate the energy requested during this time step (kWh)
            discharge_energy_requested = Q_discharge_requested * self.dt / 3600.0
            actual_discharge_energy = min(discharge_energy_requested, self.storage_energy)

            # Update the PCM storage by subtracting the discharged energy.
            self.storage_energy -= actual_discharge_energy

            # Convert the actual discharged energy back to power (kW) over the time step.
            Q_cool = actual_discharge_energy * 3600.0 / self.dt

            # In discharge mode, assume the HP is off (zero consumption)
            energy_consumed = 0.0

            # Reset the charging timer if switching from a charging cycle.
            self.charging_time = 0.0

            EER = 0.0  # No EER in discharging mode

        # Create the observation dictionary with current state and outputs.
        observation = {
            'storage_energy': self.storage_energy,      # kWh
            'charging_time': self.charging_time,     # seconds
            'Q_cool': Q_cool,         # kW cooling delivered in this step
            'energy_consumed': energy_consumed,  # kWh energy used in this step
            'EER': EER      # Energy Efficiency Ratio
        }
        return observation

    def reset(self, initial_storage=32.0):
        """
        Reset the simulation to an initial state.

        Parameters:
            initial_storage (float): Initial stored energy in kWh.

        Returns:
            state (dict): The initial state.
        """
        self.storage_energy = initial_storage
        self.charging_time = 0.0
        return {
            'storage_energy': self.storage_energy,
            'charging_time': self.charging_time
        }


# ===== Example usage =====
if __name__ == "__main__":
    # Create an instance with a 60-second time step.
    system = PCMHeatPumpSystem(dt=900, initial_storage=32.0)

    # Example action for normal operation (no PCM intervention)
    action_normal = {
        'rpm': 2500,        # example rpm value
        'T_cond': 40,       # example condenser temperature in °C
        'pcm_mode': 0       # near zero -> normal operation
    }

    # Step in normal mode
    obs = system.step(action_normal)
    print("Normal operation observation:", obs)

    # Example action for discharging mode (negative pcm_mode)
    action_discharging = {
        'pcm_mode': 1,
        'Q_discharge': 5    # request 5 kW discharge cooling power
    }
    obs = system.step(action_discharging)
    print("Discharging mode observation:", obs)
