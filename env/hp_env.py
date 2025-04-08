# Given coefficients for cooling capacity Q_dot_cool
Q_intercept = -0.2071864002858792
a = 0.005333811621408454      # coefficient for rpm
b = -0.1365379343864041        # coefficient for T_cond
c = -7.001133469771883e-07      # coefficient for rpm^2
d = 0.0012415247870363416       # coefficient for T_cond^2

# Coefficients for the HP EER correlation (unitless)
EER_intercept = 4.421297840489518
e = 0.009091070826281583       # coefficient for rpm
f = -0.38184371581722676       # coefficient for T_cond
g = -1.8923608336165108e-06     # coefficient for rpm^2
h = 0.003420273670974565       # coefficient for T_cond^2


def hp_env(rpm, T_cond, dt):
    Q_dot_cool_single = (
        Q_intercept + a * rpm + b * T_cond +
        c * rpm**2 + d * T_cond**2
    )

    Q_dot_cool = Q_dot_cool_single * 5

    EER = (
        EER_intercept + e * rpm + f * T_cond +
        g * rpm**2 + h * T_cond**2
    )
    e_hp = Q_dot_cool / EER * (dt / 3600)
    # elif rpm < 1200:
    #     Q_dot_cool = 0
    #     EER = 0
    #     e_hp = 0

    # elif rpm > 2900:
    #     rpm = 2900
    #     Q_dot_cool_single = (
    #         Q_intercept + a * rpm + b * T_cond +
    #         c * rpm**2 + d * T_cond**2
    #     )

        # Q_dot_cool = Q_dot_cool_single * 5

        # EER = (
        #     EER_intercept + e * rpm + f * T_cond +
        #     g * rpm**2 + h * T_cond**2
        # )
        # e_hp = Q_dot_cool / EER

    return Q_dot_cool, EER, e_hp


def pcm_env(Q_dot_pcm, dt, soc_init):
    Q_pcm = Q_dot_pcm * (dt / 3600)
    soc_final = soc_init + Q_pcm

    if soc_final > 1:
        soc_final = 1

    elif soc_final < 0:
        soc_final = 0

    return soc_final
