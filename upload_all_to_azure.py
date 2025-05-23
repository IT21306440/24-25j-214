import pandas as pd
from sqlalchemy import create_engine

# ✅ Azure SQL full connection string with URL-encoded password (@ → %40)
engine = create_engine(
    "mssql+pyodbc://solaradmin:Solar%40123456@solar-server.database.windows.net:1433/solar_project_db?driver=ODBC+Driver+17+for+SQL+Server"
)

# ✅ Map of local CSV file paths to SQL table names
file_table_map = {
    r"C:\Users\94772\Documents\Latest Research Full System\data\charging_vs_discharging_comparison.csv":
        'charging_vs_discharging_comparison',

    r"C:\Users\94772\Documents\Latest Research Full System\data\merged_forecast_2025_2027.csv":
        'merged_forecast_2025_2027',

    r"C:\Users\94772\Documents\Latest Research Full System\data\NewRandunifinal_combined_solar_irradiance.csv":
        'NewRandunifinal_combined_solar_irradiance',

    r"C:\Users\94772\Documents\Latest Research Full System\data\optimized_daily_power_release1.csv":
        'optimized_daily_power_release1',

    r"C:\Users\94772\Documents\Latest Research Full System\data\optimized_hourly_power_release.csv":
        'optimized_hourly_power_release',

    r"C:\Users\94772\Documents\Latest Research Full System\data\top_3_ranked_solar_locations.csv":
        'top_3_ranked_solar_locations',

    r"C:\Users\94772\Documents\Latest Research Full System\data\total_forecasted_solar_power_mwNew.csv":
        'total_forecasted_solar_power_mwNew',

    r"C:\Users\94772\Documents\Latest Research Full System\data\New folder\hybrid_lstm_xgb_2yr_forecast for Day Peak.csv":
        'hybrid_lstm_xgb_2yr_forecast_for_Day_Peak',

    r"C:\Users\94772\Documents\Latest Research Full System\data\New folder\hybrid_lstm_xgb_3yr_forecast_for_Off_Peak.csv":
        'hybrid_lstm_xgb_3yr_forecast_for_Off_Peak',

    r"C:\Users\94772\Documents\Latest Research Full System\data\New folder\hybrid_lstm_xgb_3yr_forecast_for_Peak1.csv":
        'hybrid_lstm_xgb_3yr_forecast_for_Peak1'
}

# ✅ Upload each CSV file
for file_path, table_name in file_table_map.items():
    try:
        print(f"🔄 Uploading '{table_name}' from:\n  {file_path}")
        df = pd.read_csv(file_path)
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print(f"✅ Success: Uploaded '{table_name}'\n")
    except Exception as e:
        print(f"❌ Failed to upload '{table_name}': {e}\n")
