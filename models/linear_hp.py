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
    T_cond_range = np.linspace(15, 35, 100)
    RPM, T_cond = np.meshgrid(rpm_range, T_cond_range)

    # Compute the original model values over the grid
    Q_orig = Q_dot_cool_original(RPM, T_cond)
    EER_orig = EER_original(RPM, T_cond)
    e_orig = Q_orig / EER_orig

    # Prepare data for multiple linear regression
    # Our independent variables are rpm and T_cond (only linear terms)
    X = np.column_stack((RPM.ravel(), T_cond.ravel()))
    y_Q = Q_orig.ravel()
    y_EER = EER_orig.ravel()
    y_e = e_orig.ravel()

    # Fit a linear regression model for Q_dot_cool
    model_Q = LinearRegression()
    model_Q.fit(X, y_Q)

    # Fit a linear regression model for EER
    model_EER = LinearRegression()
    model_EER.fit(X, y_EER)

    # Fit a linear regression model for e
    model_e = LinearRegression()
    model_e.fit(X, y_e)

    return model_Q, model_EER, model_e


model_Q, model_EER, model_e = linear_hp()


def hp_predict(rpm, T_cond):
    # Retrieve pre-trained linear models
    Q_pred = (model_Q.intercept_ +
              model_Q.coef_[0] * rpm +
              model_Q.coef_[1] * T_cond)

    EER_pred = (model_EER.intercept_ +
                model_EER.coef_[0] * rpm +
                model_EER.coef_[1] * T_cond)

    e_pred = (model_e.intercept_ +
              model_e.coef_[0] * rpm +
              model_e.coef_[1] * T_cond)

    return Q_pred, EER_pred, e_pred


def hp_invert(Q, T_cond):
    """
    Given a desired cooling capacity Q and condensing temperature T_cond,
    this function inverts the linear regression model for Q_dot_cool to compute
    the required rpm, and then uses this rpm to predict EER and e.

    Parameters:
        Q (float or array-like): Cooling capacity (Q_dot_cool) value(s).
        T_cond (float or array-like): Condensing temperature value(s).

    Returns:
        rpm: Calculated rpm value(s) that would produce the desired Q at T_cond.
        EER: Predicted EER based on the calculated rpm and T_cond.
        e: Predicted e based on the calculated rpm and T_cond.
    """
    # Ensure that the coefficient for rpm in the Q_dot_cool model is non-zero.
    if model_Q.coef_[0] == 0:
        raise ValueError("The coefficient for rpm is zero; cannot invert the model.")

    Q_real = Q * 0.2

    # Invert the linear model for Q_dot_cool:
    # Q = intercept + coef_rpm * rpm + coef_T * T_cond   =>   rpm = (Q - intercept - coef_T * T_cond) / coef_rpm
    rpm = (Q_real - model_Q.intercept_ - model_Q.coef_[1] * T_cond) / model_Q.coef_[0]

    # Compute the corresponding EER and e using the other models:
    EER = model_EER.intercept_ + model_EER.coef_[0] * rpm + model_EER.coef_[1] * T_cond
    e = model_e.intercept_ + model_e.coef_[0] * rpm + model_e.coef_[1] * T_cond

    return rpm, EER, e


if __name__ == "__main__":
    rpm, EER, e = hp_invert(5, 35)
    print(f"rpm: {rpm}, EER: {EER}, e: {e}")


'''
    # Test the hp_predict function
    rpm = 2900
    T_cond = 35
    Q_pred, EER_pred, e_pred = hp_predict(rpm, T_cond)
    print(f"Q_pred: {Q_pred}, EER_pred: {EER_pred}, e_pred: {e_pred}")
'''


'''
    model_Q, model_EER, model_e = linear_hp()
    rpm_range = np.linspace(1200, 2900, 100)
    T_cond_range = np.linspace(10, 35, 100)
    RPM, T_cond = np.meshgrid(rpm_range, T_cond_range)

    # Compute the original model values over the grid
    Q_orig = Q_dot_cool_original(RPM, T_cond)
    EER_orig = EER_original(RPM, T_cond)
    e_orig = Q_orig / EER_orig

    # Prepare data for multiple linear regression
    # Our independent variables are rpm and T_cond (only linear terms)
    X = np.column_stack((RPM.ravel(), T_cond.ravel()))
    y_Q = Q_orig.ravel()
    y_EER = EER_orig.ravel()
    y_e = e_orig.ravel()

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

    # Fit a linear regression model for e
    model_e = LinearRegression()
    model_e.fit(X, y_e)
    y_e_pred = model_e.predict(X)
    e_pred = y_e_pred.reshape(RPM.shape)

    # Calculate performance metrics for Q_dot_cool
    r2_Q = r2_score(y_Q, y_Q_pred)
    rmse_Q = root_mean_squared_error(y_Q, y_Q_pred)

    # Calculate performance metrics for EER
    r2_EER = r2_score(y_EER, y_EER_pred)
    rmse_EER = root_mean_squared_error(y_EER, y_EER_pred)

    # Calculate performance metrics for e
    r2_e = r2_score(y_e, y_e_pred)
    rmse_e = root_mean_squared_error(y_e, y_e_pred)

    print("Q_dot_cool Linear Regression Metrics:")
    print("  R^2 Score:", r2_Q)
    print("  RMSE Score:", rmse_Q)
    print("")
    print("EER Linear Regression Metrics:")
    print("  R^2 Score:", r2_EER)
    print("  RMSE Score:", rmse_EER)
    print("")
    print("e Linear Regression Metrics:")
    print("  R^2 Score:", r2_e)
    print("  RMSE Score:", rmse_e)
'''
