class hp_system(object):
    def __init__(self, dt=900.0):
        """
        Initialize the PCM thermal storage-HP system.

        Parameters:
            dt (float): Time step in seconds.
            initial_storage (float): Initial stored energy in kWh.
        """
        self.dt = dt  # time step in seconds

        # Coefficients for the HP cooling capacity correlation
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

    def compute_Q_dot_cool(self, rpm, T_cond):
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
                - 'rpm' (float): Compressor rpm.
                - 'T_cond' (float): Condenser temperature.

        Returns:
            observation (dict): Contains updated state and outputs:
                - 'Q_dot_cool' (kW): Cooling power delivered in this step.
                - 'e_hp' (kWh): Electricity consumed by the heat pump
                  in this step.
        """

        # Initialize outputs
        Q_dot_cool = 0.0              # Cooling power delivered in kW
        e_hp = 0.0     # HP energy consumption in kWh
        Q_cool = 0.0    # HP energy output (kWh) in this time step
        EER = 0.0  # default initialization

        rpm = action.get('rpm', 0)
        T_cond = action.get('T_cond', 0)
        Q_dot_cool = self.compute_Q_dot_cool(rpm, T_cond)
        EER = self.compute_EER(rpm, T_cond)

        # Cooling energy delivered during the time step (kWh)
        Q_cool = Q_dot_cool * self.dt / 3600.0
        # Electricity consumed by the heat pump
        e_hp = Q_cool / EER

        # Create the observation dictionary with current state and outputs.
        observation = {
            'Q_dot_cool': Q_dot_cool,   # kW cooling delivered in this step
            'Q_cool': Q_cool,  # Cooling energy delivered in this step (kWh)
            'e_hp': e_hp,   # Electricity consumed in this step
            'EER': EER      # Energy Efficiency Ratio
        }

        return observation


class pcm_system(object):
    def __init__(self, dt=900.0, SoC=27.0):
        """
        Initialize the PCM thermal storage.

        Parameters:
            dt (float): Time step in seconds.
            initial_storage (float): Initial stored energy in kWh.
        """
        self.dt = dt  # time step in seconds
        self.SoC = SoC  # state of charge in kWh

    def step_disc(self, action):
        """
        Advance the system by one time step.

        Parameters:
            action (dict): Dictionary of control inputs. Expected keys:
                - 'Q_discharge' (float): Discharge cooling power in kW.

        Returns:
            observation (dict): Contains updated state and outputs:
                - 'SoC' (float): State of charge in kWh.
        """
        Q_dot_discharge = action.get('Q_dot_discharge', 0)
        self.SoC -= Q_dot_discharge * self.dt / 3600.0

        observation = {
            'SoC': self.SoC,  # state of charge in kWh
            'Q_dot_discharge': Q_dot_discharge,  # Discharge power in kW
            'Q_discharge': (Q_dot_discharge * self.dt / 3600.0)
        }

        return observation


# ===== Example usage =====
if __name__ == "__main__":
    # Create instances
    hp = hp_system(dt=900)          # Heat Pump instance
    pcm = pcm_system(dt=900, SoC=27.0)  # PCM storage fully charged (27 kWh)

    # Step 1: Normal HP operation without PCM (cooling directly from HP)
    action_hp = {'rpm': 2500, 'T_cond': 40}
    hp_obs = hp.step(action_hp)

    print("HP operation observation (without PCM):")
    print(hp_obs)

    # Step 2: Discharge PCM storage to meet cooling demand
    action_pcm_discharge = {'Q_dot_discharge': 5}  # 5 kW discharge cooling
    pcm_obs = pcm.step_disc(action_pcm_discharge)

    print("PCM discharge observation:")
    print(pcm_obs)
