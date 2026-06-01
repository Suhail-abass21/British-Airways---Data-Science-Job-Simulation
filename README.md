British Airways Data Science Simulation - Customer Booking Prediction

 Project Overview

This project was completed as part of the **British Airways Data Science Job Simulation** on Forage. The goal was to predict whether a customer will complete an airline booking using historical booking data.

 Key Objectives

- Clean and explore 50,000+ customer booking records
- Engineer meaningful features to improve model performance
- Train a Random Forest classifier to predict booking completion
- Identify top drivers influencing customer booking behavior
- Deliver data-driven recommendations to increase conversion rates

 Dataset

The dataset contains 50,000 records with 14 features including:
- `num_passengers` - Number of passengers in booking
- `purchase_lead` - Days between purchase and flight
- `length_of_stay` - Duration of trip
- `route` - Flight origin-destination pair
- `booking_origin` - Customer's country of origin
- `wants_extra_baggage` - Customer opted for extra baggage
- `wants_preferred_seat` - Customer selected preferred seat
- `wants_in_flight_meals` - Customer pre-ordered meals
- `booking_complete` - **Target variable** (1 = completed, 0 = not completed)

 Feature Engineering

Created 8+ new features to capture hidden patterns:

| Feature | Description |
|---------|-------------|
| `flight_day_encoded` | Ordinal encoding of day of week |
| `is_weekend` | Weekend flight indicator |
| `is_red_eye` | Red-eye flight indicator (<6am or >9pm) |
| `total_wants` | Count of extra services requested |
| `lead_time_bin` | Binned purchase lead time |
| `stay_bin` | Binned length of stay |
| `lead_per_passenger` | Lead time per passenger ratio |
| `route_frequency` | Frequency encoding for route popularity |
| `origin_frequency` | Frequency encoding for booking origin |

 Model

Algorithm: Random Forest Classifier

Key Parameters:
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)
