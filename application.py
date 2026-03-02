from flask import Flask, render_template # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.preprocessing import PolynomialFeatures # type: ignore
from data_preprocessing import prepare_ticket_data

app = Flask(__name__)

def generate_narrative(historical_counts, predicted_counts):
    baseline_avg = sum(historical_counts[-7:]) / 7
    
    max_pred = max(predicted_counts)
    max_index = predicted_counts.index(max_pred)
  
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    spike_day = days_of_week[max_index % 7] 
    
    percent_increase = ((max_pred - baseline_avg) / baseline_avg) * 100
    
    if percent_increase >= 30:
        return f"⚠️ Alert: Projected {int(percent_increase)}% increase in software escalations next {spike_day}. Recommendation: Allocate 2 additional Tier-2 support specialists to the morning shift to reduce user friction."
    elif percent_increase >= 15:
        return f"⚠️ Notice: Projected {int(percent_increase)}% increase in tickets next {spike_day}. Recommendation: Monitor the Tier-1 queue closely to prevent bottlenecks."
    else:
        return "✅ Operations Normal: Predicted volumes are within standard thresholds. No additional staffing required."

@app.route('/')
def dashboard():
    df = prepare_ticket_data('helpdesk_tickets.csv')
    historical_counts = df['Total_Tickets'].tolist()
    current_days_count = len(historical_counts)
    
    X = np.array(range(current_days_count)).reshape(-1, 1)
    y = df['Total_Tickets'].values
    
    poly_feat = PolynomialFeatures(degree=4)
    X_poly = poly_feat.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    
    next_days = np.array(range(current_days_count, current_days_count + 7)).reshape(-1, 1)
    next_poly = poly_feat.transform(next_days)
    predictions = poly_reg.predict(next_poly)
    predicted_counts = [int(p) for p in predictions]
    
    insight_message = generate_narrative(historical_counts, predicted_counts)

    total_days = current_days_count + 7
    labels = [f"Day {i+1}" for i in range(total_days)]
    
    historical_data_for_chart = historical_counts + [None] * 7
    
    predicted_data_for_chart = [None] * (current_days_count - 1)
    predicted_data_for_chart.append(historical_counts[-1]) # The connecting point
    predicted_data_for_chart.extend(predicted_counts)
    
    return render_template('index.html', 
                           labels=labels, 
                           historical=historical_data_for_chart, 
                           predicted=predicted_data_for_chart,
                           insight_message=insight_message) 

if __name__ == '__main__':
    app.run(debug=True)