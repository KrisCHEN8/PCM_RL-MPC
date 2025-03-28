import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error


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


# Original quadratic models for Q_dot_cool and EER
def Q_dot_cool_original(rpm, T_cond):
    return Q_intercept + a*rpm + b*T_cond + c*rpm**2 + d*T_cond**2


def EER_original(rpm, T_cond):
    return EER_intercept + e*rpm + f*T_cond + g*rpm**2 + h*T_cond**2


def linear_hp():
    # Define a grid of operating conditions
    # Let's choose rpm between 1200 and 2900, and T_cond between 10°C and 35°C
    rpm_range = np.linspace(1200, 2900, 100)
    T_cond_range = np.linspace(10, 35, 100)
    RPM, T_cond = np.meshgrid(rpm_range, T_cond_range)

    # Compute the original model values over the grid
    Q_orig = Q_dot_cool_original(RPM, T_cond)
    EER_orig = EER_original(RPM, T_cond)

    # Prepare data for multiple linear regression
    # Our independent variables are rpm and T_cond (only linear terms)
    X = np.column_stack((RPM.ravel(), T_cond.ravel()))
    y_Q = Q_orig.ravel()
    y_EER = EER_orig.ravel()

    # Fit a linear regression model for Q_dot_cool
    model_Q = LinearRegression()
    model_Q.fit(X, y_Q)

    # Fit a linear regression model for EER
    model_EER = LinearRegression()
    model_EER.fit(X, y_EER)

    return model_Q, model_EER


def hp_predict(rpm, T_cond):
    model_Q, model_EER = linear_hp()
    Q_pred = model_Q.predict(np.column_stack((rpm, T_cond)))
    EER_pred = model_EER.predict(np.column_stack((rpm, T_cond)))

    return Q_pred, EER_pred


if __name__ == "__main__":
    model_Q, model_EER = linear_hp()
    rpm_range = np.linspace(1200, 2900, 100)
    T_cond_range = np.linspace(10, 35, 100)
    RPM, T_cond = np.meshgrid(rpm_range, T_cond_range)

    # Compute the original model values over the grid
    Q_orig = Q_dot_cool_original(RPM, T_cond)
    EER_orig = EER_original(RPM, T_cond)

    # Prepare data for multiple linear regression
    # Our independent variables are rpm and T_cond (only linear terms)
    X = np.column_stack((RPM.ravel(), T_cond.ravel()))
    y_Q = Q_orig.ravel()
    y_EER = EER_orig.ravel()

    # Fit a linear regression model for Q_dot_cool
    model_Q = LinearRegression()
    model_Q.fit(X, y_Q)
    y_Q_pred = model_Q.predict(X)
    Q_pred = y_Q_pred.reshape(RPM.shape)

    # Fit a linear regression model for EER
    model_EER = LinearRegression()
    model_EER.fit(X, y_EER)
    y_EER_pred = model_EER.predict(X)
    EER_pred = y_EER_pred.reshape(RPM.shape)

    # Calculate performance metrics for Q_dot_cool
    r2_Q = r2_score(y_Q, y_Q_pred)
    rmse_Q = root_mean_squared_error(y_Q, y_Q_pred)

    # Calculate performance metrics for EER
    r2_EER = r2_score(y_EER, y_EER_pred)
    rmse_EER = root_mean_squared_error(y_EER, y_EER_pred)

    print("Q_dot_cool Linear Regression Metrics:")
    print("  R^2 Score:", r2_Q)
    print("  RMSE Score:", rmse_Q)
    print("")
    print("EER Linear Regression Metrics:")
    print("  R^2 Score:", r2_EER)
    print("  RMSE Score:", rmse_EER)
