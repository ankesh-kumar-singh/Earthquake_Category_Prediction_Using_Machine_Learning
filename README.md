# Earthquake Prediction App

A Streamlit-based Machine Learning application that analyzes recent global earthquake data and predicts the earthquake magnitude category using spatial, temporal, and geological features.  
The application also simulates possible earthquake occurrences near a user-defined location over the next 7 days.

Important Disclaimer  
This project is intended only for educational and research purposes. Earthquakes cannot be reliably predicted using current scientific methods. The results produced by this application must not be used for disaster warning or life-critical decisions.

---

## Key Features

- Live Earthquake Data  
  Automatically fetches monthly global earthquake data from the USGS Earthquake Hazards Program

- Machine Learning Classification  
  Predicts earthquake magnitude category using a Random Forest Classifier

- Imbalanced Dataset Handling  
  Uses SMOTE (Synthetic Minority Over-sampling Technique)

- Location-Based Prediction  
  User inputs latitude, longitude, depth, date, and time

- 7-Day Future Risk Simulation  
  Simulates multiple depths, nearby coordinates, and time intervals

- Distance Filtering  
  Displays predicted events within 1000 km of the user’s location

- Interactive Maps  
  Streamlit map and PyDeck scatter plot

- Modern User Interface  
  Custom CSS styling with clean layout

---

## Earthquake Magnitude Classification

| Category   | Magnitude Range |
|-----------|----------------|
| Micro     | < 2.0          |
| Minor     | 2.0 – 4.0      |
| Light     | 4.0 – 5.0      |
| Moderate  | 5.0 – 6.0      |
| Strong    | 6.0 – 7.0      |
| Major     | 7.0 – 8.0      |
| Great     | ≥ 8.0          |

---

## Machine Learning Workflow

1. Data Collection  
   Monthly earthquake CSV feed from USGS

2. Data Preprocessing  
   - Removal of missing values  
   - Timestamp decomposition into:
     - Year
     - Month
     - Day
     - Hour
     - Minute
     - Second

3. Feature Set  
   - Latitude  
   - Longitude  
   - Depth (km)  
   - Temporal features  

4. Target Variable  
   Earthquake magnitude category

5. Train-Test Split  
   80% training, 20% testing

6. Class Imbalance Handling  
   Dynamic SMOTE sampling

7. Model  
   Random Forest Classifier with 100 estimators

8. Evaluation  
   - Accuracy score  
   - Classification report  

---

## Future Earthquake Simulation Logic

- Simulates:
  - Next 7 days
  - Depth variations (±30 km)
  - Nearby latitude and longitude offsets
  - Time intervals every 6 hours

- Filters predictions based on:
  - Magnitude category ≥ Moderate
  - Prediction confidence ≥ 0.5
  - Distance ≤ 1000 km

- Outputs:
  - Date and time
  - Latitude and longitude
  - Depth
  - Predicted category
  - Confidence score
  - Class probabilities

---

## Visualizations

- Recent earthquake locations using Streamlit map
- Magnitude-based scatter plot using PyDeck
- Tabular display of prediction probabilities
- Preview of recent earthquake records

---

## Technology Stack

Programming Language  
- Python 3.8+

Libraries and Frameworks  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- GeoPy  
- PyDeck  
- Requests  

Data Source  
- USGS Earthquake Hazards Program

---

