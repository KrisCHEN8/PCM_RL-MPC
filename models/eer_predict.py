import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Given coefficients for the HP EER correlation (unitless)
EER_intercept = 4.421297840489518
e = 0.009091070826281583       # coefficient for rpm
f = -0.38184371581722676       # coefficient for T_cond
g = -1.8923608336165108e-06     # coefficient for rpm^2
h = 0.003420273670974565       # coefficient for T_cond^2


# Original quadratic model for EER
def EER_original(rpm, T_cond):
    return EER_intercept + e*rpm + f*T_cond + g*rpm**2 + h*T_cond**2


# Fix rpm to a constant value
rpm_fixed = 2000

# Generate a range of T_cond values (for example, from 30°C to 60°C)
T_cond_values = np.linspace(15, 35, 100).reshape(-1, 1)
# Compute the corresponding EER values using the original quadratic model
# Use .ravel() to convert the result to a 1D array for regression
EER_values = EER_original(rpm_fixed, T_cond_values).ravel()

# Fit a linear regression model using only T_cond as predictor
model = LinearRegression()
model.fit(T_cond_values, EER_values)

# Generate predictions using the linear model
EER_pred = model.predict(T_cond_values)

# Calculate performance metrics: R^2 and RMSE
r2 = r2_score(EER_values, EER_pred)
rmse = np.sqrt(mean_squared_error(EER_values, EER_pred))


def eer_predict(T_cond):
    eer_predicted = model.intercept_ + model.coef_[0] * T_cond

    return eer_predicted


if __name__ == "__main__":
    # Display the linear model parameters and performance metrics
    T_cond = 28
    eer_predicted = eer_predict(T_cond)
    print(eer_predicted)
