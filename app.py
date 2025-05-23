from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from pathlib import Path
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from collections import OrderedDict

app = Flask(__name__)

# Get the directory where the script (app.py) is located
BASE_DIR = Path(__file__).resolve().parent

# Define paths relative to the script's directory
DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "efficientnet_v2_l_dust_detector.pth"
GRAPHS_FOLDER = BASE_DIR / "static" / "inverter_fault_graphs"

# Function to Load CSV with Safe Error Handling
def load_csv(filename):
    filepath = DATA_PATH / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        # Convert Date column if it exists
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    else:
        print(f"Warning: {filename} not found at {filepath}")
        return None

# Load Data
from sqlalchemy import create_engine

# Azure SQL Connection
engine = create_engine(
    "mssql+pyodbc://solaradmin:Solar%40123456@solar-server.database.windows.net:1433/solar_project_db?driver=ODBC+Driver+17+for+SQL+Server"
)

# Load DataFrames from Azure SQL (same variable names preserved)
top_3_locations_df = pd.read_sql("SELECT * FROM top_3_ranked_solar_locations", con=engine)
forecast_results_df = pd.read_sql("SELECT * FROM merged_forecast_2025_2027", con=engine)
electricity_demand_df = None  # No table provided
charging_vs_discharging_df = pd.read_sql("SELECT * FROM charging_vs_discharging_comparison", con=engine)
battery_revenue_df = pd.read_sql("SELECT * FROM charging_vs_discharging_comparison", con=engine)
optimized_daily_power_df = pd.read_sql("SELECT * FROM optimized_daily_power_release1", con=engine)
optimized_hourly_power_df = pd.read_sql("SELECT * FROM optimized_hourly_power_release", con=engine)
demand_forecast_df = pd.read_sql("SELECT * FROM hybrid_lstm_xgb_2yr_forecast_for_Day_Peak", con=engine)
peak_demand_df = pd.read_sql("SELECT * FROM hybrid_lstm_xgb_3yr_forecast_for_Peak1", con=engine)
day_peak_demand_df = pd.read_sql("SELECT * FROM hybrid_lstm_xgb_2yr_forecast_for_Day_Peak", con=engine)
off_peak_demand_df = pd.read_sql("SELECT * FROM hybrid_lstm_xgb_3yr_forecast_for_Off_Peak", con=engine)
future_solar_irradiance_df = pd.read_sql("SELECT * FROM NewRandunifinal_combined_solar_irradiance", con=engine)
forecasted_solar_power_df = pd.read_sql("SELECT * FROM total_forecasted_solar_power_mwNew", con=engine)

# Ensure `available_years` is properly initialized
available_years = sorted(top_3_locations_df["year"].unique().tolist()) if top_3_locations_df is not None else []

@app.route('/')
def index():
    return render_template('index.html', years=available_years)

@app.route('/battery')
def battery():
    return render_template('battery.html')

@app.route('/integrated')
def integrated():
    return render_template('integrated.html')

@app.route('/demand')
def demand():
    return render_template('demand.html')

@app.route('/maintainance')
def maintainance():
    return render_template('maintainance.html')

@app.route("/dust")
def dust():
    return render_template("dust.html")

@app.route("/upload")
def upload_page():
    return render_template("dust.html")

@app.route('/get_available_locations', methods=['POST'])
def get_available_locations():
    if forecast_results_df is not None and "Location" in forecast_results_df.columns:
        locations = sorted(forecast_results_df["Location"].unique().tolist())
        return jsonify({"locations": locations})
    return jsonify({"error": "No locations available"}), 500

@app.route('/get_months', methods=['POST'])
def get_available_months():
    year = int(request.form.get('year'))
    if forecast_results_df is not None and "Date" in forecast_results_df.columns:
        months = forecast_results_df[forecast_results_df["Date"].dt.year == year]["Date"].dt.strftime("%B").unique().tolist()
        return jsonify({"months": months})
    return jsonify({"error": "Forecast data not available"}), 500

@app.route('/get_available_years', methods=['POST'])
def get_available_years():
    if top_3_locations_df is not None:
        years = sorted(top_3_locations_df["year"].unique().tolist())
    else:
        years = []
    return jsonify({"years": years})

@app.route('/get_future_locations', methods=['POST'])
def get_future_locations():
    if future_solar_irradiance_df is None:
        return jsonify({"error": "Future solar irradiance data not available"}), 500

    if "Location" not in future_solar_irradiance_df.columns:
        return jsonify({"error": "Column 'Location' missing in future_solar_irradiance dataset"}), 500

    locations = sorted(future_solar_irradiance_df["Location"].dropna().unique().tolist())
    
    if not locations:
        return jsonify({"error": "No future locations available"}), 500
    
    return jsonify({"locations": locations})

@app.route('/get_best_location', methods=['POST'])
def get_best_location():
    year = request.form.get('year')

    if not year:
        return jsonify({"error": "Year parameter is missing"}), 400

    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Invalid year format"}), 400

    location_coordinates = {
        "siyabalanduwa": {"lat": 6.8102, "lon": 81.5861},
        "vavuniya": {"lat": 8.7515, "lon": 80.4976},
        "mannar": {"lat": 8.9776, "lon": 79.9045},
        "trincomali": {"lat": 8.5874, "lon": 81.2152},
        "kilinochchi": {"lat": 9.3961, "lon": 80.3982},
        "polonnaruwa": {"lat": 7.9403, "lon": 81.0188},
        "hambantota": {"lat": 6.1246, "lon": 81.1185},
        "matara": {"lat": 5.9549, "lon": 80.5519},
        "monaragala": {"lat": 6.8724, "lon": 81.3487},
        "munalthivu_puththalam": {"lat": 8.0362, "lon": 79.8406},
    }

    if top_3_locations_df is None or top_3_locations_df.empty:
        return jsonify({"error": "Top 3 locations data not available"}), 500

    # Filter for the selected year
    top_3_row = top_3_locations_df[top_3_locations_df["year"] == year]
    if top_3_row.empty:
        return jsonify({"error": f"No top 3 locations found for {year}"}), 404

    # Extract top 3 locations and their irradiance values
    row = top_3_row.iloc[0]
    top_locations = [
        {
            "rank": 1,
            "location": row["best_location1"].lower(),
            "irradiance": row["irradiance_best_location1"],
            "latitude": location_coordinates.get(row["best_location1"].lower(), {}).get("lat"),
            "longitude": location_coordinates.get(row["best_location1"].lower(), {}).get("lon")
        },
        {
            "rank": 2,
            "location": row["best_location2"].lower(),
            "irradiance": row["irradiance_best_location2"],
            "latitude": location_coordinates.get(row["best_location2"].lower(), {}).get("lat"),
            "longitude": location_coordinates.get(row["best_location2"].lower(), {}).get("lon")
        },
        {
            "rank": 3,
            "location": row["best_location3"].lower(),
            "irradiance": row["irradiance_best_location3"],
            "latitude": location_coordinates.get(row["best_location3"].lower(), {}).get("lat"),
            "longitude": location_coordinates.get(row["best_location3"].lower(), {}).get("lon")
        }
    ]

    # Check for missing coordinates
    for loc in top_locations:
        if loc["latitude"] is None or loc["longitude"] is None:
            return jsonify({"error": f"Coordinates not found for {loc['location']}"}), 500

    return jsonify({"top_locations": top_locations})

@app.route('/get_irradiance_curve', methods=['POST'])
def get_irradiance_curve():
    year = request.form.get('year')
    month = request.form.get('month')
    time_frame = request.form.get('time_frame', 'daily')

    if not year or not time_frame:
        return jsonify({"error": "Missing year or time frame parameter"}), 400

    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Invalid year format"}), 400

    if top_3_locations_df is None or future_solar_irradiance_df is None:
        return jsonify({"error": "Required datasets not loaded"}), 500

    top_3_row = top_3_locations_df[top_3_locations_df["year"] == year]
    if top_3_row.empty:
        return jsonify({"error": f"No top 3 locations found for {year}"}), 404

    best_location = top_3_row.iloc[0]["best_location1"].lower()

    required_columns = ["Date", "Forecasted Solar Irradiance", "Location"]
    for col in required_columns:
        if col not in future_solar_irradiance_df.columns:
            return jsonify({"error": f"Missing column '{col}' in future_solar_irradiance dataset"}), 500

    future_solar_irradiance_df["Date"] = pd.to_datetime(future_solar_irradiance_df["Date"], errors="coerce")
    future_solar_irradiance_df["Location"] = future_solar_irradiance_df["Location"].str.lower()
    future_solar_irradiance_df["Year"] = future_solar_irradiance_df["Date"].dt.year
    future_solar_irradiance_df["Month"] = future_solar_irradiance_df["Date"].dt.strftime("%B")

    if time_frame == "monthly":
        filtered_data = future_solar_irradiance_df[
            (future_solar_irradiance_df["Year"] == year) &
            (future_solar_irradiance_df["Location"] == best_location)
        ]
        if filtered_data.empty:
            return jsonify({"error": f"No irradiance data found for {best_location} in {year}"}), 404
        filtered_data = filtered_data.groupby(filtered_data["Date"].dt.month)["Forecasted Solar Irradiance"].mean().reset_index()
        filtered_data["Month"] = filtered_data["Date"].map({
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        })
        filtered_data = filtered_data.sort_values("Date")
        return jsonify({
            "year": year,
            "location": best_location,
            "dates": filtered_data["Month"].tolist(),
            "irradiance": filtered_data["Forecasted Solar Irradiance"].tolist()
        })
    else:
        if not month:
            return jsonify({"error": "Month parameter required for daily time frame"}), 400
        filtered_data = future_solar_irradiance_df[
            (future_solar_irradiance_df["Year"] == year) &
            (future_solar_irradiance_df["Month"] == month) &
            (future_solar_irradiance_df["Location"] == best_location)
        ]
        if filtered_data.empty:
            return jsonify({"error": f"No irradiance data found for {best_location} in {month} {year}"}), 404
        filtered_data = filtered_data.sort_values("Date")
        return jsonify({
            "year": year,
            "month": month,
            "location": best_location,
            "dates": filtered_data["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "irradiance": filtered_data["Forecasted Solar Irradiance"].tolist()
        })

@app.route('/get_solar_irradiance', methods=['POST'])
def get_solar_irradiance():
    year = request.form.get('year')
    month = request.form.get('month')
    location = request.form.get('location')
    time_frame = request.form.get('time_frame', 'daily')

    if not year or not location or not time_frame:
        return jsonify({"error": "Missing year, location, or time frame parameter"}), 400

    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Invalid year format"}), 400

    if forecast_results_df is None:
        return jsonify({"error": "Solar irradiance data not available"}), 500

    if "Date" not in forecast_results_df.columns or "Forecasted Solar Irradiance" not in forecast_results_df.columns or "Location" not in forecast_results_df.columns:
        return jsonify({"error": "Required columns missing in dataset"}), 500

    forecast_results_df["Date"] = pd.to_datetime(forecast_results_df["Date"], errors="coerce")
    forecast_results_df["Location"] = forecast_results_df["Location"].str.lower().str.strip()
    location = location.lower().strip()

    if time_frame == "monthly":
        filtered_data = forecast_results_df[
            (forecast_results_df["Date"].dt.year == year) &
            (forecast_results_df["Location"] == location)
        ]
        if filtered_data.empty:
            return jsonify({"error": f"No data found for {location} in {year}"}), 404
        filtered_data = filtered_data.groupby(filtered_data["Date"].dt.month)["Forecasted Solar Irradiance"].mean().reset_index()
        filtered_data["Month"] = filtered_data["Date"].map({
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        })
        filtered_data = filtered_data.sort_values("Date")
        return jsonify({
            "dates": filtered_data["Month"].tolist(),
            "irradiance": filtered_data["Forecasted Solar Irradiance"].tolist()
        })
    else:
        if not month:
            return jsonify({"error": "Month parameter required for daily time frame"}), 400
        filtered_data = forecast_results_df[
            (forecast_results_df["Date"].dt.year == year) &
            (forecast_results_df["Date"].dt.strftime("%B") == month) &
            (forecast_results_df["Location"] == location)
        ]
        if filtered_data.empty:
            return jsonify({"error": f"No data found for {location} in {month} {year}"}), 404
        filtered_data = filtered_data.sort_values("Date")
        return jsonify({
            "dates": filtered_data["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "irradiance": filtered_data["Forecasted Solar Irradiance"].tolist()
        })

@app.route('/get_future_solar_irradiance', methods=['POST'])
def get_future_solar_irradiance():
    year = request.form.get('year')
    month = request.form.get('month')
    location = request.form.get('location')
    time_frame = request.form.get('time_frame', 'daily')

    if not year or not location:
        return jsonify({"error": "Missing year or location parameter"}), 400

    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Invalid year format"}), 400

    if future_solar_irradiance_df is None:
        return jsonify({"error": "Future solar irradiance data not available"}), 500

    required_columns = ["Date", "Forecasted Solar Irradiance", "Location"]
    missing_columns = [col for col in required_columns if col not in future_solar_irradiance_df.columns]
    if missing_columns:
        return jsonify({"error": f"Missing columns in dataset: {missing_columns}"}), 500

    future_solar_irradiance_df["Date"] = pd.to_datetime(future_solar_irradiance_df["Date"], errors="coerce")
    future_solar_irradiance_df["Location"] = future_solar_irradiance_df["Location"].str.lower()
    future_solar_irradiance_df["Year"] = future_solar_irradiance_df["Date"].dt.year
    future_solar_irradiance_df["Month"] = future_solar_irradiance_df["Date"].dt.strftime("%B")

    if time_frame == "monthly":
        filtered_data = future_solar_irradiance_df[
            (future_solar_irradiance_df["Year"] == year) &
            (future_solar_irradiance_df["Location"] == location)
        ]
        if filtered_data.empty:
            return jsonify({"error": f"No data found for {location} in {year}"}), 404
        filtered_data = filtered_data.groupby(filtered_data["Date"].dt.month)["Forecasted Solar Irradiance"].mean().reset_index()
        filtered_data["Month"] = filtered_data["Date"].map({
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        })
        filtered_data = filtered_data.sort_values("Date")
        return jsonify({
            "year": year,
            "location": location,
            "dates": filtered_data["Month"].tolist(),
            "forecasted_irradiance": filtered_data["Forecasted Solar Irradiance"].tolist()
        })
    else:
        if not month:
            return jsonify({"error": "Month parameter required for daily time frame"}), 400
        filtered_data = future_solar_irradiance_df[
            (future_solar_irradiance_df["Year"] == year) &
            (future_solar_irradiance_df["Month"] == month) &
            (future_solar_irradiance_df["Location"] == location)
        ]
        if filtered_data.empty:
            return jsonify({"error": f"No data found for {location} in {month} {year}"}), 404
        filtered_data = filtered_data.sort_values("Date")
        return jsonify({
            "year": year,
            "month": month,
            "location": location,
            "dates": filtered_data["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "forecasted_irradiance": filtered_data["Forecasted Solar Irradiance"].tolist()
        })

@app.route('/get_forecasted_solar_power', methods=['POST'])
def get_forecasted_solar_power():
    year = request.form.get('year')
    month = request.form.get('month')
    time_frame = request.form.get('time_frame', 'daily')

    if not year:
        return jsonify({"error": "Missing year parameter"}), 400

    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Invalid year format" }), 400

    if forecasted_solar_power_df is None:
        return jsonify({"error": "Forecasted solar power data not available"}), 500

    if "Date" not in forecasted_solar_power_df.columns:
        return jsonify({"error": "Missing 'Date' column in dataset"}), 500

    forecasted_solar_power_df["Date"] = pd.to_datetime(forecasted_solar_power_df["Date"], errors="coerce")
    forecasted_solar_power_df["Year"] = forecasted_solar_power_df["Date"].dt.year
    forecasted_solar_power_df["Month"] = forecasted_solar_power_df["Date"].dt.strftime("%B")

    if time_frame == "monthly":
        filtered_data = forecasted_solar_power_df[
            (forecasted_solar_power_df["Year"] == year)
        ]
        if filtered_data.empty:
            return jsonify({"error": f"No forecasted solar power data available for {year}"}), 404
        filtered_data = filtered_data.groupby(filtered_data["Date"].dt.month)["forecasted_solar_power_mw"].sum().reset_index()
        filtered_data["Month"] = filtered_data["Date"].map({
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        })
        filtered_data = filtered_data.sort_values("Date")
        return jsonify({
            "dates": filtered_data["Month"].tolist(),
            "solar_power": filtered_data["forecasted_solar_power_mw"].tolist()
        })
    else:
        if not month:
            return jsonify({"error": "Month parameter required for daily time frame"}), 400
        filtered_data = forecasted_solar_power_df[
            (forecasted_solar_power_df["Year"] == year) & 
            (forecasted_solar_power_df["Month"] == month)
        ]
        if filtered_data.empty:
            return jsonify({"error": f"No forecasted solar power data available for {month} {year}"}), 404
        filtered_data = filtered_data.sort_values("Date")
        return jsonify({
            "dates": filtered_data["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "solar_power": filtered_data["forecasted_solar_power_mw"].tolist()
        })

@app.route('/get_all_peak_demand', methods=['POST'])
def get_all_peak_demand():
    year = request.form.get('year')
    month = request.form.get('month')
    time_frame = request.form.get('time_frame', 'daily')

    if not year:
        return jsonify({"error": "Year is required"}), 400

    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Invalid year format"}), 400

    month_mapping = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }

    for df in [peak_demand_df, day_peak_demand_df, off_peak_demand_df]:
        if df is not None:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if time_frame == "monthly":
        peak_filtered = peak_demand_df[peak_demand_df["Date"].dt.year == year] if peak_demand_df is not None else pd.DataFrame()
        day_peak_filtered = day_peak_demand_df[day_peak_demand_df["Date"].dt.year == year] if day_peak_demand_df is not None else pd.DataFrame()
        off_peak_filtered = off_peak_demand_df[off_peak_demand_df["Date"].dt.year == year] if off_peak_demand_df is not None else pd.DataFrame()

        if peak_filtered.empty or day_peak_filtered.empty or off_peak_filtered.empty:
            return jsonify({"error": f"No demand data available for {year}"}), 404

        peak_filtered = peak_filtered.groupby(peak_demand_df["Date"].dt.month)[peak_demand_df.columns[1]].mean().reset_index()
        peak_filtered["Month"] = peak_filtered["Date"].map({
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        })
        day_peak_filtered = day_peak_filtered.groupby(day_peak_demand_df["Date"].dt.month)[day_peak_demand_df.columns[1]].mean().reset_index()
        day_peak_filtered["Month"] = day_peak_filtered["Date"].map({
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        })
        off_peak_filtered = off_peak_filtered.groupby(off_peak_demand_df["Date"].dt.month)[off_peak_demand_df.columns[1]].mean().reset_index()
        off_peak_filtered["Month"] = off_peak_filtered["Date"].map({
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        })

        peak_filtered = peak_filtered.sort_values("Date")
        day_peak_filtered = day_peak_filtered.sort_values("Date")
        off_peak_filtered = off_peak_filtered.sort_values("Date")

        return jsonify({
            "dates": peak_filtered["Month"].tolist(),
            "peak_demand": peak_filtered[peak_demand_df.columns[1]].tolist(),
            "day_peak_demand": day_peak_filtered[day_peak_demand_df.columns[1]].tolist(),
            "off_peak_demand": off_peak_filtered[off_peak_demand_df.columns[1]].tolist(),
        })
    else:
        if not month:
            return jsonify({"error": "Month parameter required for daily time frame"}), 400
        month_num = month_mapping.get(month)
        if month_num is None:
            return jsonify({"error": "Invalid month"}), 400
        peak_filtered = peak_demand_df[
            (peak_demand_df["Date"].dt.year == year) & (peak_demand_df["Date"].dt.month == month_num)
        ] if peak_demand_df is not None else pd.DataFrame()
        day_peak_filtered = day_peak_demand_df[
            (day_peak_demand_df["Date"].dt.year == year) & (day_peak_demand_df["Date"].dt.month == month_num)
        ] if day_peak_demand_df is not None else pd.DataFrame()
        off_peak_filtered = off_peak_demand_df[
            (off_peak_demand_df["Date"].dt.year == year) & (off_peak_demand_df["Date"].dt.month == month_num)
        ] if off_peak_demand_df is not None else pd.DataFrame()
        if peak_filtered.empty or day_peak_filtered.empty or off_peak_filtered.empty:
            return jsonify({"error": f"No demand data available for {month} {year}"}), 404
        peak_filtered = peak_filtered.sort_values("Date")
        day_peak_filtered = day_peak_filtered.sort_values("Date")
        off_peak_filtered = off_peak_filtered.sort_values("Date")
        return jsonify({
            "dates": peak_filtered["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "peak_demand": peak_filtered.iloc[:, 1].tolist(),
            "day_peak_demand": day_peak_filtered.iloc[:, 1].tolist(),
            "off_peak_demand": off_peak_filtered.iloc[:, 1].tolist(),
        })

@app.route('/get_power_data', methods=['POST'])
def get_power_data():
    global optimized_daily_power_df

    date_str = request.form.get('date')
    time_frame = request.form.get('time_frame', 'daily')

    if not date_str or not time_frame:
        return jsonify({"error": "Missing date or time frame parameter"}), 400

    if optimized_daily_power_df is None:
        return jsonify({"error": "Power data not available"}), 500

    required_columns = ["Date", "Optimized_Daily_MW"]
    missing_columns = [col for col in required_columns if col not in optimized_daily_power_df.columns]
    if missing_columns:
        return jsonify({"error": f"Required columns missing in dataset: {missing_columns}"}), 500

    optimized_daily_power_df["Date"] = pd.to_datetime(optimized_daily_power_df["Date"], errors="coerce")

    try:
        selected_date = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
        if pd.isnull(selected_date):
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid date format. Expected format: YYYY-MM-DD"}), 400

    if time_frame == "monthly":
        filtered_data = optimized_daily_power_df[optimized_daily_power_df["Date"].dt.year == selected_date.year]
        if filtered_data.empty:
            return jsonify({"error": f"No data available for {selected_date.year}"}), 404
        filtered_data = filtered_data.groupby(filtered_data["Date"].dt.month)["Optimized_Daily_MW"].sum().reset_index()
        filtered_data["Month"] = filtered_data["Date"].map({
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        })
        filtered_data = filtered_data.sort_values("Date")
    else:
        filtered_data = optimized_daily_power_df[
            (optimized_daily_power_df["Date"].dt.year == selected_date.year) & 
            (optimized_daily_power_df["Date"].dt.month == selected_date.month)
        ]
        if filtered_data.empty:
            return jsonify({"error": f"No data available for {selected_date.strftime('%Y-%m')}"}), 404
        filtered_data = filtered_data.groupby(filtered_data["Date"].dt.date)["Optimized_Daily_MW"].sum().reset_index()
        filtered_data["Date"] = filtered_data["Date"].astype(str)

    return jsonify({
        "dates": filtered_data["Month"].tolist() if time_frame == "monthly" else filtered_data["Date"].tolist(),
        "power_release": filtered_data["Optimized_Daily_MW"].tolist()
    })

@app.route('/get_optimized_daily_power', methods=['POST'])
def get_optimized_daily_power():
    global optimized_daily_power_df

    month = request.form.get('month', 'All')

    if optimized_daily_power_df is None:
        return jsonify({"error": "Daily power release data not available"}), 500

    if "Day" in optimized_daily_power_df.columns and "Date" not in optimized_daily_power_df.columns:
        optimized_daily_power_df = optimized_daily_power_df.rename(columns={"Day": "Date"})

    if not pd.api.types.is_datetime64_any_dtype(optimized_daily_power_df["Date"]):
        optimized_daily_power_df["Date"] = pd.to_datetime(optimized_daily_power_df["Date"], errors="coerce")

    optimized_daily_power_df = optimized_daily_power_df.dropna(subset=["Date"])

    optimized_daily_power_df["Month"] = optimized_daily_power_df["Date"].dt.strftime("%B")

    filtered_data = optimized_daily_power_df if month.lower() == "all" else \
        optimized_daily_power_df[optimized_daily_power_df["Month"].str.lower() == month.lower()]

    if filtered_data.empty:
        return jsonify({"error": f"No data available for {month}"}), 404

    return jsonify({
        "dates": filtered_data["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "power_release": filtered_data["Optimized_Daily_MW"].tolist()
    })

@app.route('/get_charging_vs_discharging', methods=['POST'])
def get_charging_vs_discharging():
    month = request.form.get('month', 'January')

    if charging_vs_discharging_df is None:
        return jsonify({"error": "Battery charging/discharging data not available"}), 500

    required_columns = ["Date", "Charging_Cost (USCts/kWh)", "Discharging_Revenue (USCts/kWh)"]
    missing_columns = [col for col in required_columns if col not in charging_vs_discharging_df.columns]

    if missing_columns:
        return jsonify({"error": f"Missing required columns: {missing_columns}"}), 500

    charging_vs_discharging_df["Date"] = pd.to_datetime(charging_vs_discharging_df["Date"], errors="coerce")
    charging_vs_discharging_df["Month"] = charging_vs_discharging_df["Date"].dt.strftime("%B")

    daily_data = charging_vs_discharging_df.groupby(charging_vs_discharging_df["Date"].dt.date).agg({
        "Charging_Cost (USCts/kWh)": "sum",
        "Discharging_Revenue (USCts/kWh)": "sum"
    }).reset_index()

    if month.lower() != "all":
        daily_data["Month"] = pd.to_datetime(daily_data["Date"]).dt.strftime("%B")
        filtered_data = daily_data[daily_data["Month"].str.lower() == month.lower()]
    else:
        filtered_data = daily_data

    if filtered_data.empty:
        print(f"No data found for {month}. Falling back to January.")
        fallback_data = daily_data[daily_data["Month"] == "January"]
        if fallback_data.empty:
            return jsonify({"dates": ["No Data"], "charging_cost": [0], "discharging_revenue": [0]})

        filtered_data = fallback_data

    return jsonify({
        "dates": filtered_data["Date"].astype(str).tolist(),
        "charging_cost": filtered_data["Charging_Cost (USCts/kWh)"].tolist(),
        "discharging_revenue": filtered_data["Discharging_Revenue (USCts/kWh)"].tolist()
    })

@app.route('/get_battery_revenue', methods=['POST'])
def get_battery_revenue():
    year = request.form.get('year')
    month = request.form.get('month', 'January')
    time_frame = request.form.get('time_frame', 'daily')

    if not year:
        return jsonify({"error": "Year parameter is missing"}), 400

    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Invalid year format"}), 400

    if battery_revenue_df is None:
        return jsonify({"error": "Battery revenue data not available"}), 500

    required_columns = ["Date", "Net_Profit (USCts/kWh)"]
    missing_columns = [col for col in required_columns if col not in battery_revenue_df.columns]

    if missing_columns:
        return jsonify({"error": f"Missing required columns: {missing_columns}"}), 500

    battery_revenue_df["Date"] = pd.to_datetime(battery_revenue_df["Date"], errors="coerce")
    battery_revenue_df["Year"] = battery_revenue_df["Date"].dt.year
    battery_revenue_df["Month"] = battery_revenue_df["Date"].dt.strftime("%B")

    if time_frame == "monthly":
        filtered_data = battery_revenue_df[battery_revenue_df["Year"] == year]
        if filtered_data.empty:
            return jsonify({"error": f"No data found for {year}"}), 404
        filtered_data = filtered_data.groupby(filtered_data["Date"].dt.month)["Net_Profit (USCts/kWh)"].sum().reset_index()
        filtered_data["Month"] = filtered_data["Date"].map({
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        })
        filtered_data = filtered_data.sort_values("Date")
    else:
        if not month:
            return jsonify({"error": "Month parameter required for daily time frame"}), 400
        filtered_data = battery_revenue_df[(battery_revenue_df["Year"] == year) & (battery_revenue_df["Month"] == month)]
        if filtered_data.empty:
            print(f"No data found for {month}. Falling back to January.")
            fallback_data = battery_revenue_df[battery_revenue_df["Month"] == "January"]
            if fallback_data.empty:
                return jsonify({"dates": ["No Data"], "revenue": [0]})
            filtered_data = fallback_data
        filtered_data = filtered_data.groupby(filtered_data["Date"].dt.date)["Net_Profit (USCts/kWh)"].sum().reset_index()
        filtered_data["Date"] = filtered_data["Date"].astype(str)

    return jsonify({
        "dates": filtered_data["Month"].tolist() if time_frame == "monthly" else filtered_data["Date"].tolist(),
        "revenue": filtered_data["Net_Profit (USCts/kWh)"].tolist()
    })

@app.route('/get_optimized_hourly_power', methods=['POST'])
def get_optimized_hourly_power():
    if optimized_hourly_power_df is None:
        return jsonify({"error": "Hourly power release data not available"}), 500

    return jsonify({
        "dates": optimized_hourly_power_df["Hour"].tolist(),
        "power_release": optimized_hourly_power_df["Optimized_Hourly_MW"].tolist()
    })

# Model Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
CLASS_NAMES = {0: "Clean", 1: "Dusty"}

# Load Model Function
def build_model():
    model_weights = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_l(weights=model_weights).to(DEVICE)

    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=1280, out_features=256, bias=True),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=NUM_CLASSES, bias=True)
    ).to(DEVICE)

    return model

# Load Model
model = build_model()
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

# Image Preprocessing
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read()))

        image_tensor = transform_image(image)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        return jsonify({
            "predicted_class": CLASS_NAMES[pred_class],
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_available_graphs():
    graphs = []
    for filename in os.listdir(GRAPHS_FOLDER):
        if filename.endswith(".png"):
            parts = filename.split("_")
            inverter = parts[1]
            date = parts[-1].replace(".png", "")
            graphs.append({"inverter": inverter, "date": date, "file": filename})
    return graphs

@app.route('/inverter', methods=['GET', 'POST'])
def inverter():
    graphs = get_available_graphs()
    inverters = sorted(set(g['inverter'] for g in graphs))
    dates = sorted(set(g['date'] for g in graphs))
    
    selected_inverter = request.form.get('inverter')
    selected_date = request.form.get('date')
    
    filtered_graphs = [g for g in graphs if (not selected_inverter or g['inverter'] == selected_inverter) and
                                          (not selected_date or g['date'] == selected_date)]
    
    return render_template('inverter.html', inverters=inverters, dates=dates, graphs=filtered_graphs, 
                           selected_inverter=selected_inverter, selected_date=selected_date)

if __name__ == '__main__':
    app.run(debug=True)