import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.preprocessing import PolynomialFeatures # type: ignore
from data_preprocessing import prepare_ticket_data

print("Fetching data: \n")
df = prepare_ticket_data('helpdesk_tickets.csv')

df['Day_Index'] = range(len(df))  #day->num

X = df['Day_Index'].values.reshape(-1, 1)  #x = day
y = df['Total_Tickets'].values             #y = ticket numbers

linear_model = LinearRegression()
linear_model.fit(X, y)

poly_converter = PolynomialFeatures(degree=4)
X_poly = poly_converter.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

current_days_count = len(df)
next_days = np.array(range(current_days_count, current_days_count + 7)).reshape(-1, 1) #today to 7 days in future

# Predict using the straight-line model
future_linear_prediction = linear_model.predict(next_days)

# Predict using the curved model
future_days_poly = poly_converter.transform(next_days)
future_poly_prediction = poly_model.predict(future_days_poly)

print("\n--- Ticket Sales for Next 7 days Predicted ---\n")
for i in range(7):
    day_num = next_days[i][0]
    print(f"Day {day_num}: Linear Predicts {int(future_linear_prediction[i])} tickets | Polynomial Predicts {int(future_poly_prediction[i])} tickets")