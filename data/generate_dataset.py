"""
Generates a synthetic air quality dataset and saves it as CSV.
Features: PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, Wind Speed
Target: AQI (Air Quality Index) + AQI Category
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 5000  # number of samples

def generate_data(n=N):
    # Pollutant concentrations (µg/m³ or ppm)
    PM25       = np.random.gamma(shape=2.5, scale=18, size=n).clip(1, 300)
    PM10       = PM25 * np.random.uniform(1.2, 2.5, n) + np.random.normal(0, 5, n)
    PM10       = PM10.clip(5, 500)
    NO2        = np.random.gamma(shape=2, scale=20, size=n).clip(5, 200)
    SO2        = np.random.gamma(shape=1.5, scale=12, size=n).clip(1, 150)
    CO         = np.random.gamma(shape=1.8, scale=0.6, size=n).clip(0.1, 10)
    O3         = np.random.normal(60, 25, n).clip(5, 200)
    Temperature = np.random.normal(25, 10, n).clip(-10, 45)
    Humidity   = np.random.uniform(20, 95, n)
    WindSpeed  = np.random.exponential(scale=3, size=n).clip(0.1, 20)

    # Add seasonal correlation
    season_effect = np.random.choice([1.0, 1.3, 0.8, 1.1], size=n)
    PM25  = (PM25 * season_effect).clip(1, 300)
    NO2   = (NO2  * season_effect).clip(5, 200)

    # AQI Calculation (simplified US EPA sub-index method)
    def aqi_pm25(c):
        # breakpoints: (Clo, Chi, ILo, IHi)
        bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
              (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500,401,500)]
        result = np.zeros_like(c)
        for clo, chi, ilo, ihi in bp:
            mask = (c >= clo) & (c <= chi)
            result[mask] = ((ihi - ilo) / (chi - clo)) * (c[mask] - clo) + ilo
        return result

    def aqi_no2(c):
        bp = [(0,53,0,50),(54,100,51,100),(101,360,101,150),
              (361,649,151,200),(650,1249,201,300),(1250,1649,301,400),(1650,2049,401,500)]
        result = np.zeros_like(c)
        for clo, chi, ilo, ihi in bp:
            mask = (c >= clo) & (c <= chi)
            result[mask] = ((ihi - ilo) / (chi - clo)) * (c[mask] - clo) + ilo
        return result

    aqi_from_pm25 = aqi_pm25(PM25)
    aqi_from_no2  = aqi_no2(NO2)
    # Overall AQI is the maximum sub-index
    AQI = np.maximum(aqi_from_pm25, aqi_from_no2)
    # Add some noise
    AQI = (AQI + np.random.normal(0, 5, n)).clip(0, 500).astype(int)

    # AQI Category
    def categorize(aqi_val):
        if aqi_val <= 50:   return "Good"
        elif aqi_val <= 100: return "Moderate"
        elif aqi_val <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi_val <= 200: return "Unhealthy"
        elif aqi_val <= 300: return "Very Unhealthy"
        else:                return "Hazardous"

    Category = np.array([categorize(v) for v in AQI])

    df = pd.DataFrame({
        "PM2.5":       np.round(PM25, 2),
        "PM10":        np.round(PM10, 2),
        "NO2":         np.round(NO2, 2),
        "SO2":         np.round(SO2, 2),
        "CO":          np.round(CO, 3),
        "O3":          np.round(O3, 2),
        "Temperature": np.round(Temperature, 1),
        "Humidity":    np.round(Humidity, 1),
        "WindSpeed":   np.round(WindSpeed, 2),
        "AQI":         AQI,
        "Category":    Category,
    })

    return df


if __name__ == "__main__":
    df = generate_data()
    out_path = os.path.join(os.path.dirname(__file__), "air_quality.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path}")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nCategory distribution:\n{df['Category'].value_counts()}")
