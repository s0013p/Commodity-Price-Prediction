import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import mysql.connector
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

from prediction import SarimaXGBoostEnsembleModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = "579b464db66ec23bdd000001c4f6e3a1106c4281715a8bd45f4197ed"
WEATHER_API_KEY = "f334e2ba66ffa5970c4207243b7b7494"
DB_CONFIG = {'host': 'localhost', 'user': 'root', 'password': 'Password', 'database': 'database1'}


class DatabaseHandler:
    def __init__(self, host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], database=DB_CONFIG['database']):
        self.host, self.user, self.password, self.database, self.connection = host, user, password, database, None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(host=self.host, user=self.user, password=self.password, database=self.database)
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                self._create_tables()
        except Exception as e:
            st.error(f"Database error: {e}")
            raise

    def _create_tables(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute("""CREATE TABLE IF NOT EXISTS commodity_data (id INT AUTO_INCREMENT PRIMARY KEY, commodity VARCHAR(100), state VARCHAR(100), district VARCHAR(100), arrival_date DATE, min_price FLOAT, modal_price FLOAT, max_price FLOAT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE KEY unique_record (commodity, state, district, arrival_date)) ENGINE=InnoDB;""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS combine_data (id INT AUTO_INCREMENT PRIMARY KEY, commodity VARCHAR(100), state VARCHAR(100), district VARCHAR(100), arrival_date DATE, year INT, min_price FLOAT, modal_price FLOAT, max_price FLOAT, supply_metric_tons FLOAT, demand_metric_tons FLOAT, consumer_price_index FLOAT, temperature FLOAT, visibility FLOAT, wind_speed FLOAT, clouds FLOAT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE KEY unique_combined_record (commodity, state, district, arrival_date)) ENGINE=InnoDB;""")
            self.connection.commit()
            cursor.close()
            logger.info("Successfully created/verified database tables")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise

    def get_unique_commodities(self):
        return ["Wheat", "Rice", "Tur(Arhar)Dal", "Potato", "Onion", "Tomato", "Soyabean", "Moong Dal", "Sugar", "Gur"]
    
    def get_states_for_commodity(self, commodity):
        return ["Andhra Pradesh", "Bihar", "Gujarat", "Haryana","Karnataka", "Madhya Pradesh", "Maharashtra", "Punjab", "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal"]

    def get_districts_for_state_commodity(self, state, commodity):
        default_districts = {
            "Andhra Pradesh": ["Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna", "Kurnool", "Prakasam", "Srikakulam", "Visakhapatnam", "West Godavari", "YSR Kadapa", "Vijayawada", "Tirupati", "Kakinada", "Guntur", "Vijayanagaram"],
            "Maharashtra": ["Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded", "Nashik", "Osmanabad", "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"]
        }
        return default_districts.get(state, ["Ahmednagar", "District2", "District3", "District4"])

    def import_csv_data(self, file_path):
        try:
            if not self.connection or not self.connection.is_connected(): self.connect()
            df = pd.read_csv(file_path)
            required_columns = ['State', 'District', 'Commodity', 'Year','Supply (Metric Tons)', 'Demand (Metric Tons)', 'Consumer_Price_Index']
            for column in required_columns:
                if column not in df.columns:
                    logger.error(f"Missing required column in CSV: {column}")
                    return f"Error: Missing column {column} in CSV file"
            logger.info(f"Successfully imported CSV with {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error importing CSV data: {str(e)}")
            return f"Error: {str(e)}"

    def save_commodity_data(self, data):
        try:
            if not self.connection or not self.connection.is_connected(): self.connect()
            cursor = self.connection.cursor()
            insert_query = """INSERT INTO commodity_data (commodity, state, district, arrival_date, min_price, max_price, modal_price)VALUES (%s, %s, %s, %s, %s, %s, %s)"""
            records_inserted = 0
            for record in data:
                arrival_date_str = record.get('Arrival_Date', '')
                try:
                    arrival_date = datetime.strptime(arrival_date_str, '%d/%m/%Y')
                except ValueError:
                    try:
                        arrival_date = datetime.strptime(arrival_date_str, '%d-%m-%Y')
                    except ValueError:
                        logger.error(f"Unable to parse date: {arrival_date_str}")
                        continue
                values = (record.get('Commodity'),record.get('State'),record.get('District'),arrival_date,float(record.get('Min_Price', 0)),float(record.get('Max_Price', 0)),float(record.get('Modal_Price', 0)))
                cursor.execute(insert_query, values)
                records_inserted += cursor.rowcount
            self.connection.commit()
            cursor.close()
            logger.info(f"Successfully saved {records_inserted} new commodity records to database")
            return records_inserted
        except Exception as e:
            logger.error(f"Error saving commodity data to database: {str(e)}")
            if self.connection: self.connection.rollback()
            raise
    
    def check_data_exists(self, commodity, state, district, start_date, end_date):
        try:
            if not self.connection or not self.connection.is_connected(): self.connect()
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM commodity_data WHERE commodity = %s AND state = %s AND district = %s AND arrival_date BETWEEN %s AND %s", (commodity, state, district, start_date, end_date))
            result = cursor.fetchone()
            cursor.close()
            count = result[0] if result else 0
            logger.info(f"Found {count} existing records for {commodity} in {district}, {state}")
            return count > 0
        except Exception as e:
            logger.error(f"Error checking existing data: {str(e)}")
            return False

    def get_commodity_data(self, commodity, state, district, start_date, end_date):
        try:
            if not self.connection or not self.connection.is_connected(): self.connect()
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT commodity, state, district, arrival_date,AVG(min_price) as min_price, AVG(max_price) as max_price, AVG(modal_price) as modal_price FROM commodity_data WHERE commodity = %s AND state = %s AND district = %s AND arrival_date BETWEEN %s AND %s GROUP BY commodity, state, district, arrival_date ORDER BY arrival_date", (commodity, state, district, start_date, end_date))
            records = cursor.fetchall()
            cursor.close()
            return records
        except Exception as e:
            logger.error(f"Error retrieving commodity data: {str(e)}")
            return []

    def save_combined_data(self, commodity_data, csv_data, weather_data=None):
        try:
            if not self.connection or not self.connection.is_connected(): self.connect()
            cursor = self.connection.cursor()
            if commodity_data:

                min_date = min(record['arrival_date'] for record in commodity_data)
                max_date = max(record['arrival_date'] for record in commodity_data)

                cursor.execute("DELETE FROM combine_data WHERE commodity = %s AND state = %s AND district = %s AND arrival_date BETWEEN %s AND %s", (commodity_data[0]['commodity'], commodity_data[0]['state'], commodity_data[0]['district'], min_date, max_date))
                self.connection.commit()
                processed_data = process_and_merge_data(commodity_data, weather_data or [], csv_data)
                if processed_data is not None:
                    insert_query = """INSERT INTO combine_data (commodity, state, district, arrival_date, year, min_price, modal_price, max_price,supply_metric_tons, demand_metric_tons, consumer_price_index,temperature, visibility, wind_speed, clouds) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                    records_inserted = 0
                    for _, row in processed_data.iterrows():
                        try:
                            values = (row.get('Commodity', ''), row.get('State', ''), row.get('District', ''), row['Arrival_Date'] if pd.notnull(row.get('Arrival_Date')) else None, int(row.get('Year', 0)) if pd.notnull(row.get('Year')) else None, float(row.get('Min_Price', 0)) if pd.notnull(row.get('Min_Price')) else 0, float(row.get('Modal_Price', 0)) if pd.notnull(row.get('Modal_Price')) else 0, float(row.get('Max_Price', 0)) if pd.notnull(row.get('Max_Price')) else 0, float(row.get('Supply (Metric Tons)', 0)) if pd.notnull(row.get('Supply (Metric Tons)')) else 0, float(row.get('Demand (Metric Tons)', 0)) if pd.notnull(row.get('Demand (Metric Tons)')) else 0, float(row.get('CPI', 100)) if pd.notnull(row.get('CPI')) else 100, float(row.get('temperature', 25.0)) if pd.notnull(row.get('temperature')) else 25.0, float(row.get('visibility', 10.0)) if pd.notnull(row.get('visibility')) else 10.0, float(row.get('wind_speed', 5.0)) if pd.notnull(row.get('wind_speed')) else 5.0, float(row.get('clouds', 50.0)) if pd.notnull(row.get('clouds')) else 50.0)
                            cursor.execute(insert_query, values)
                            records_inserted += cursor.rowcount
                        except Exception as row_error:
                            logger.error(f"Error inserting row: {row_error}")
                            continue
                    self.connection.commit()
                    cursor.close()
                    logger.info(f"Successfully saved {records_inserted} combined records with weather data")
                    return records_inserted
        except Exception as e:
            logger.error(f"Error saving combined data: {str(e)}")
            if self.connection: self.connection.rollback()
            raise

    def get_combined_data(self, commodity, state, district, start_date, end_date):
        try:
            if not self.connection or not self.connection.is_connected(): self.connect()
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT commodity, state, district, arrival_date, year, min_price, modal_price, max_price, supply_metric_tons, demand_metric_tons, consumer_price_index, temperature, visibility, wind_speed, clouds FROM combine_data WHERE commodity = %s AND state = %s AND district = %s AND arrival_date BETWEEN %s AND %s ORDER BY arrival_date", (commodity, state, district, start_date, end_date))
            records = cursor.fetchall()
            cursor.close()
            return records
        except Exception as e:
            logger.error(f"Error retrieving combined data: {str(e)}")
            return []


class CommodityDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"

    def fetch_data_in_range(self, state, district, commodity, start_date, end_date):
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Date parsing error: {str(e)}")
            st.error(f"Date parsing error: {str(e)}")
            return []
        all_records = []
        current_date = start
        total_days = (end - start).days + 1
        progress_bar = st.progress(0)
        status_text = st.empty()
        days_processed = 0
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%d-%m-%Y')
            except ValueError:
                logger.error(f"Date conversion error for {date_str}")
                current_date += timedelta(days=1)
                days_processed += 1
                continue
            params = {"api-key": self.api_key, "format": "json", "filters[State.keyword]": state, "filters[District.keyword]": district, "filters[Commodity.keyword]": commodity, "filters[Arrival_Date]": formatted_date, "offset": 0, "limit": 1000}
            try:
                response = requests.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                daily_records = data.get("records", [])
                all_records.extend(daily_records)
                logger.info(f"Fetched {len(daily_records)} records for {date_str}")
                days_processed += 1
                progress = days_processed / total_days
                progress_bar.progress(progress)
                status_text.text(f"Processing {date_str}: {len(daily_records)} records found")
                current_date += timedelta(days=1)
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for date {date_str}: {str(e)}")
                current_date += timedelta(days=1)
                days_processed += 1
                continue
        progress_bar.empty()
        status_text.empty()
        logger.info(f"Total records fetched: {len(all_records)}")
        return all_records

    def get_combined_data_with_all_sources(self, commodity, state, district, start_date, end_date):
        try:
            if not self.connection or not self.connection.is_connected(): self.connect()
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT commodity, state, district, arrival_date, year, min_price, modal_price, max_price, supply_metric_tons, demand_metric_tons, consumer_price_index FROM combine_data WHERE commodity = %s AND state = %s AND district = %s AND arrival_date BETWEEN %s AND %s ORDER BY arrival_date", (commodity, state, district, start_date, end_date))
            records = cursor.fetchall()
            cursor.close()
            return records
        except Exception as e:
            logger.error(f"Error retrieving combined data: {str(e)}")
            return []


class WeatherDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def fetch_weather_data(self, city, date):
        try:
            params = {'q': city, 'appid': self.api_key, 'units': 'metric'}
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return {'city': city, 'temperature': data['main']['temp'], 'visibility': data.get('visibility', 0), 'wind_speed': data['wind']['speed'], 'clouds': data.get('clouds', {}).get('all', 0), 'country': data['sys']['country'], 'fetch_date': date}
        except Exception as e:
            st.error(f"Weather data fetch error: {e}")
            return None

    def fetch_weather_data_in_range(self, city, start_date, end_date):
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Date parsing error: {str(e)}")
            return []
        weather_data = []
        current_date = start
        while current_date <= end:
            data = self.fetch_weather_data(city, current_date.strftime('%Y-%m-%d'))
            if data: weather_data.append(data)
            current_date += timedelta(days=1)
        return weather_data


class SupplyDemandLoader:
    def __init__(self, supply_demand_file_path):
        self.supply_demand_file_path = supply_demand_file_path
        self.supply_demand_data = None

    def load_data(self):
        try:
            self.supply_demand_data = pd.read_csv(self.supply_demand_file_path)
            self.supply_demand_data['Year'] = self.supply_demand_data['Year'].astype(str)
            for col in ['State', 'District', 'Commodity']:
                self.supply_demand_data[col] = self.supply_demand_data[col].str.strip()
            logger.info("Supply and demand data loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading supply and demand data: {e}")
            return False


def process_and_merge_data(price_records, weather_data, supply_demand_loader):
    if not price_records:
        logger.warning("No price records available for processing")
        return None
    df_prices = pd.DataFrame(price_records)
    df_prices['Arrival_Date'] = pd.to_datetime(df_prices['Arrival_Date'], errors='coerce', format='mixed', dayfirst=True)
    df_prices = df_prices.dropna(subset=['Arrival_Date'])
    for col in ['State', 'District', 'Commodity']:
        if col in df_prices.columns: df_prices[col] = df_prices[col].str.strip()
    df_prices['Year'] = df_prices['Arrival_Date'].dt.year.astype(str)
    df_combined = df_prices.copy()
    if weather_data:
        try:
            df_weather = pd.DataFrame(weather_data)
            df_weather['fetch_date'] = pd.to_datetime(df_weather['fetch_date'], errors='coerce')
            df_weather = df_weather.dropna(subset=['fetch_date'])
            if not df_weather.empty:
                df_combined['merge_date'] = df_combined['Arrival_Date'].dt.date
                df_weather['merge_date'] = df_weather['fetch_date'].dt.date
                df_combined = pd.merge(df_combined, df_weather[['merge_date', 'temperature', 'visibility', 'wind_speed', 'clouds']], on='merge_date', how='left')
                df_combined = df_combined.drop(columns=['merge_date'])
                logger.info(f"Successfully merged weather data for {len(df_weather)} dates")
            else:
                logger.warning("Weather data is empty after processing")
                for col in ['temperature', 'visibility', 'wind_speed', 'clouds']: df_combined[col] = None
        except Exception as e:
            logger.error(f"Error processing weather data: {e}")
            for col in ['temperature', 'visibility', 'wind_speed', 'clouds']: df_combined[col] = None
    else:
        for col in ['temperature', 'visibility', 'wind_speed', 'clouds']: df_combined[col] = None
    if supply_demand_loader and hasattr(supply_demand_loader, 'supply_demand_data') and supply_demand_loader.supply_demand_data is not None:
        supply_demand_data = supply_demand_loader.supply_demand_data.copy()
        for col in ['State', 'District', 'Commodity']:
            if col in supply_demand_data.columns: supply_demand_data[col] = supply_demand_data[col].str.strip()
        merge_cols = ['State', 'District', 'Commodity', 'Year']
        df_combined = pd.merge(df_combined, supply_demand_data[merge_cols + ['Supply (Metric Tons)', 'Demand (Metric Tons)', 'Consumer_Price_Index']], on=merge_cols, how='left')
        df_combined.rename(columns={'Consumer_Price_Index': 'CPI'}, inplace=True)
    else:
        df_combined['Supply (Metric Tons)'] = 0
        df_combined['Demand (Metric Tons)'] = 0
        df_combined['CPI'] = 100
    numeric_fillna_map = {'temperature': 25.0, 'visibility': 10.0, 'wind_speed': 5.0, 'clouds': 50.0, 'Supply (Metric Tons)': 0, 'Demand (Metric Tons)': 0, 'CPI': 100}
    for col, default_value in numeric_fillna_map.items():
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].fillna(default_value)
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce').fillna(default_value)
    logger.info(f"Processed {len(df_combined)} records successfully")
    return df_combined



def display_price_analysis(df, combined=False):
    if df.empty:
        st.warning("No data available for analysis")
        return
    price_columns = ['min_price', 'max_price', 'modal_price']
    for col in price_columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    if combined:
        for col in ['supply_metric_tons', 'demand_metric_tons', 'consumer_price_index']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in ['temperature', 'visibility', 'wind_speed', 'clouds']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    st.subheader("📊 Price Statistics")
    st.dataframe(df[price_columns].describe().style.format("{:.2f}"))
    st.subheader("Price Trends")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['arrival_date'], y=df['min_price'], mode='lines', name='Minimum Price'))
    fig.add_trace(go.Scatter(x=df['arrival_date'], y=df['modal_price'], mode='lines', name='Modal Price', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df['arrival_date'], y=df['max_price'], mode='lines', name='Maximum Price'))
    fig.update_layout(title="Price Trends over Time", xaxis_title="Date", yaxis_title="Price (₹)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    if combined and 'consumer_price_index' in df.columns:
        st.subheader("Correlation Analysis")
        if len(df) > 1:
            correlation_cols = ['modal_price', 'consumer_price_index', 'supply_metric_tons', 'demand_metric_tons']
            correlation_df = df[correlation_cols].corr()
            fig = go.Figure(data=go.Heatmap(z=correlation_df.values, x=correlation_df.columns, y=correlation_df.index, colorscale='Viridis', zmin=-1, zmax=1))
            fig.update_layout(title='Correlation Matrix', height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.info("**Correlation Interpretation:** Values close to 1: Strong positive correlation, Values close to -1: Strong negative correlation, Values close to 0: Little to no correlation")
    st.subheader("Raw Data")
    page_size = 10
    total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
    page_number = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    display_df = df.iloc[start_idx:end_idx].copy()
    display_df['arrival_date'] = display_df['arrival_date'].dt.strftime('%Y-%m-%d')
    st.dataframe(display_df)
    st.download_button("Download Data as CSV", data=df.to_csv(index=False), file_name="commodity_data.csv", mime="text/csv")



def prediction_page():
    st.title("🎯Commodity Price Prediction")
    
    try:
        db_handler = DatabaseHandler()
        db_handler.connect()

        col1, col2 = st.columns(2)
        with col1:
            commodities = db_handler.get_unique_commodities()
            commodity = st.selectbox('Select Commodity', commodities)
            states = db_handler.get_states_for_commodity(commodity)
            state = st.selectbox('Select State', states)

        with col2:
            districts = db_handler.get_districts_for_state_commodity(state, commodity)
            district = st.selectbox('Select District', districts)
            prediction_date = st.date_input(
                'Select Date for Prediction',
                value=datetime.now(),
                min_value=datetime.now() - timedelta(days=365 * 25),
                max_value=datetime.now() + timedelta(days=365)
            )

        # Set confidence level directly in code - fixed at 95%
        confidence_level = 95

        perform_backtesting = st.checkbox("Perform Backtesting", value=True)

        if st.button('Generate Prediction'):
            with st.spinner('Processing data and generating predictions...'):
                try:
                    # Calculate date range for 5 years of historical data
                    start_date = prediction_date - timedelta(days=365 * 5)
                    end_date = prediction_date
                    
                    # Fetch historical data from combine_data table
                    historical_data = db_handler.get_commodity_data(
                        commodity=commodity,
                        state=state,
                        district=district,
                        start_date=start_date,
                        end_date=end_date
                    )


                    # Convert to DataFrame and prepare for model training
                    df = pd.DataFrame(historical_data)
                    
                    # Rename columns to match expected format
                    df = df.rename(columns={
                        'arrival_date': 'Arrival_Date',
                        'modal_price': 'Modal_Price',
                        'min_price': 'Min_Price',
                        'max_price': 'Max_Price',
                        'commodity': 'Commodity',
                        'state': 'State',
                        'district': 'District'
                    })
                    
                    # Ensure Arrival_Date is datetime
                    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
                    df = df.sort_values('Arrival_Date')
                    
                    # Check if we have enough data
                    if len(df) < 30:
                        st.warning(f"Insufficient data for prediction. Found {len(df)} records, but need at least 30 records for reliable predictions.")
                        return
                    
                    st.info(f"Using {len(df)} historical records from {df['Arrival_Date'].min().strftime('%Y-%m-%d')} to {df['Arrival_Date'].max().strftime('%Y-%m-%d')}")

                    # Initialize and train model
                    model = SarimaXGBoostEnsembleModel()
                    with st.spinner('Training model... This may take a minute.'):
                        success = model.train(df)
                        if not success:
                            st.error("Failed to train model. Please try another commodity or location.")
                            return

                    # Perform backtesting if selected
                    if perform_backtesting:
                        with st.spinner('Performing backtesting...'):
                            results = model.backtesting(df)
                            if results:
                                st.subheader("Backtesting Results")
                                
                                # Display metrics
                                metrics = results['metrics']
                                cols = st.columns(4)
                                # cols[0].metric("RMSE", f"{metrics['rmse']:.2f}%")
                                # cols[1].metric("MAE", f"{metrics['mae']:.2f}%")
                                cols[0].metric("R² Score", f"{metrics['r2']:.3f}")
                                cols[1].metric("MAPE", f"{metrics['mape']:.2f}%")
                                
                                # Plot results
                                st.subheader("Backtesting Visualization")
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=results['test_dates'],
                                    y=results['actual_prices'],
                                    name='Actual Prices',
                                    line=dict(color='blue', width=2)
                                ))
                                fig.add_trace(go.Scatter(
                                    x=results['test_dates'],
                                    y=results['ensemble_predictions'],
                                    name='Predicted Prices',
                                    line=dict(color='red', width=2)
                                ))
                                
                                fig.update_layout(
                                    title='Model Performance on Test Data',
                                    xaxis_title='Date',
                                    yaxis_title='Price (₹)',
                                    hovermode='x unified',
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    # Generate future predictions
                    with st.spinner('Generating price predictions...'):
                        predictions = model.predict(prediction_date.strftime('%Y-%m-%d'))
                        
                        if predictions.empty:
                            st.error("Failed to generate predictions. Please try another date or commodity.")
                            return
                        
                        # Ensure predictions have proper columns
                        required_columns = ['date', 'predicted_price', 'lower_bound', 'upper_bound']
                        missing_columns = [col for col in required_columns if col not in predictions.columns]
                        
                        if missing_columns:
                            st.error(f"Missing required columns in predictions: {missing_columns}")
                            st.write("Available columns:", predictions.columns.tolist())
                            st.write("Preview of predictions dataframe:", predictions.head())
                            return
                        
                        # Display predicted prices for the next 5 days
                        st.markdown(f"### Predicted Prices for {commodity} in {district}, {state}")
                        
                        # Create a table to display the predictions
                        # price_data = {
                        #     "Date": [date.strftime('%d %b %Y') for date in predictions['date']],
                        #     "Predicted Price (₹)": [f"₹{price:.2f}" for price in predictions['predicted_price']],
                        #     f"Price Range ({confidence_level}% CI)": [
                        #         f"₹{lower:.2f} - ₹{upper:.2f}" 
                        #         for lower, upper in zip(predictions['lower_bound'], predictions['upper_bound'])
                        #     ]
                        # }
                        
                        # Display the prediction table
                        # st.table(pd.DataFrame(price_data))
                        
                        # Create prediction cards with ranges
                        st.markdown("### Price Prediction")
                        cols = st.columns(5)
                        for idx, (_, row) in enumerate(predictions.iterrows()):
                            with cols[idx]:
                                st.markdown(f"""
                                <div style='padding: 1em; background-color: white; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center;'>
                                    <p style='color: #666; font-size: 0.9em;'>{row['date'].strftime('%d %b %Y')}</p>
                                    <h3 style='color: #2c3e50; margin: 0;'>₹{row['predicted_price']:.2f}</h3>
                                    <p style='color: #666; font-size: 0.8em;'>Range: ₹{row['lower_bound']:.2f} - ₹{row['upper_bound']:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Application error: {str(e)}")
    finally:
        try:
            if 'db_handler' in locals() and hasattr(db_handler, 'connection') and db_handler.connection:
                db_handler.connection.close()
        except:
            pass



def analytics_page():
    st.header("📈 Analytics")

    try:
        db_handler = DatabaseHandler()
        db_handler.connect()

        # Custom styling for the selection boxes and titles
        st.markdown("""
        <style>
        div[data-baseweb="select"] {
            background-color: rgba(30, 30, 40, 0.5);
            border-radius: 8px;
            border: 1px solid rgba(70, 70, 80, 0.5);
        }
        .selection-title {
            font-weight: bold;
            margin-bottom: 5px;
            color: white;
        }
        .stDateInput > div > div {
            background-color: rgba(30, 30, 40, 0.5);
            border-radius: 8px;
            border: 1px solid rgba(70, 70, 80, 0.5);
        }
        </style>
        """, unsafe_allow_html=True)

        # Updated tabs with enhanced data processing
        tab1, tab2 = st.tabs(["Fetch Data", "Analyze Existing Data"])
        
        # Common selection boxes for all tabs
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="selection-title">Select Commodity</div>', unsafe_allow_html=True)
                commodities = db_handler.get_unique_commodities()
                commodity = st.selectbox('Commodity', commodities, key="commodity_selectbox", label_visibility="collapsed")
                
                st.markdown('<div class="selection-title">Select State</div>', unsafe_allow_html=True)
                states = db_handler.get_states_for_commodity(commodity)
                state = st.selectbox('State', states, key="state_selectbox", label_visibility="collapsed")

            with col2:
                st.markdown('<div class="selection-title">Select District</div>', unsafe_allow_html=True)
                districts = db_handler.get_districts_for_state_commodity(state, commodity)
                district = st.selectbox('District', districts, key="district_selectbox", label_visibility="collapsed")
                
                # Date input in a row
                date_col1, date_col2 = st.columns(2)
                
                min_date = datetime.now() - timedelta(days=365*15)
                max_date = datetime.now()
                
                with date_col1:
                    st.markdown('<div class="selection-title">Start Date</div>', unsafe_allow_html=True)
                    start_date = st.date_input('Start Date', value=(max_date - timedelta(days=30)), min_value=min_date, max_value=max_date, key="start_date", label_visibility="collapsed")
                
                with date_col2:
                    st.markdown('<div class="selection-title">End Date</div>', unsafe_allow_html=True)
                    end_date = st.date_input('End Date', value=max_date, min_value=start_date, max_value=max_date, key="end_date", label_visibility="collapsed")

            if start_date > end_date:
                st.error("Start date must be before end date")
                return

        # Tab 1: Enhanced Fetch Data
        with tab1:
            # Add some spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Fetch Data button only in the Fetch New Data tab
            fetch_button = st.button('Fetch Data from API', type="primary", use_container_width=True)
            
            if fetch_button:
                # First check if data already exists for these parameters
                data_exists = db_handler.check_data_exists(
                    commodity=commodity,
                    state=state,
                    district=district,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data_exists:
                    st.warning(f"Data already exists in database for {commodity} in {district}, {state} between {start_date} and {end_date}.")
                    
                    # Show existing data count
                    existing_records = db_handler.get_commodity_data(
                        commodity=commodity,
                        state=state,
                        district=district,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if existing_records:
                        st.info(f"Found {len(existing_records)} existing records in database.")
                        
                else:
                    st.info("No existing data found. Fetching fresh data from API...")
                    
                    with st.spinner("Fetching data from API..."):
                        try:
                            data_fetcher = CommodityDataFetcher(API_KEY)
                            api_data = data_fetcher.fetch_data_in_range(
                                state=state,
                                district=district,
                                commodity=commodity,
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d')
                            )
                            
                            if api_data:
                                # Save to commodity_data table only
                                records_inserted = db_handler.save_commodity_data(api_data)
                                
                                if records_inserted > 0:
                                    st.success(f"✅ Successfully fetched and saved {records_inserted} new records to database")
                                    
                                    # Get the newly saved records to display
                                    records = db_handler.get_commodity_data(
                                        commodity=commodity,
                                        state=state,
                                        district=district,
                                        start_date=start_date,
                                        end_date=end_date
                                    )
                                    
                                    if records:
                                        df = pd.DataFrame(records)
                                        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
                                        
                                        # Display summary statistics
                                        st.subheader("📊 Fetch Summary")
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Records Fetched", len(df))
                                        with col2:
                                            st.metric("Date Range", f"{len(df['arrival_date'].dt.date.unique())} days")
                                        with col3:
                                            st.metric("Avg Modal Price", f"₹{df['modal_price'].mean():.2f}")
                                        with col4:
                                            st.metric("Price Range", f"₹{df['modal_price'].min():.0f}-₹{df['modal_price'].max():.0f}")
                                        
                                        # Offer to download the data
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            "📥 Download Fetched Data as CSV", 
                                            data=csv, 
                                            file_name=f"fetched_commodity_data_{commodity}_{district}_{start_date}_{end_date}.csv", 
                                            mime="text/csv"
                                        )
                                        
                                        # Display a preview of the data
                                        st.subheader("📋 Data Preview (Latest 10 Records)")
                                        preview_df = df.sort_values('arrival_date', ascending=False).head(10)
                                        st.dataframe(preview_df[['arrival_date', 'commodity', 'state', 'district', 'min_price', 'modal_price', 'max_price']])
                                else:
                                    st.warning("⚠️ No new records were saved. Data might already exist or API returned duplicate records.")
                            else:
                                st.warning("❌ No data found for the selected criteria from the API")
                                st.info("This could be due to:")
                                st.write("• No data available for the selected date range")
                                st.write("• API connectivity issues")
                                st.write("• Invalid commodity/location combination")
                                
                        except Exception as e:
                            st.error(f"❌ Error during data fetching: {str(e)}")
                            logger.error(f"Error in fetch data tab: {str()}")

                            
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            data_source = st.radio(
                "Select Data Source",
                ["API Data Only", "Combined Data"],
                horizontal=True
            )
            
            retrieve_button = st.button('Retrieve and Analyze Data', type="primary", use_container_width=True)
            
            if retrieve_button:
                with st.spinner("Retrieving data for analysis..."):
                    if data_source == "API Data Only":
                        data = db_handler.get_commodity_data(
                            commodity=commodity,
                            state=state,
                            district=district,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if data:
                            df = pd.DataFrame(data)
                            df['arrival_date'] = pd.to_datetime(df['arrival_date'])
                            display_price_analysis(df, combined=False)
                        else:
                            st.warning("No API data found in database for the selected criteria")
                            
                    else:
                        # Combined Data processing - fetch and merge all three data sources
                        # First get commodity data
                        commodity_data = db_handler.get_commodity_data(
                            commodity=commodity,
                            state=state,
                            district=district,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if not commodity_data:
                            st.warning("No commodity data found in database for the selected criteria. Please fetch data first in the 'Fetch Data' tab.")
                        else:
                            with st.spinner("Processing and merging all data sources..."):
                                try:
                                    # Convert commodity data to proper format for processing
                                    commodity_records = []
                                    for record in commodity_data:
                                        commodity_records.append({
                                            'Commodity': record['commodity'],
                                            'State': record['state'],
                                            'District': record['district'],
                                            'Arrival_Date': record['arrival_date'].strftime('%d/%m/%Y') if hasattr(record['arrival_date'], 'strftime') else str(record['arrival_date']),
                                            'Min_Price': record['min_price'],
                                            'Max_Price': record['max_price'],
                                            'Modal_Price': record['modal_price']
                                        })
                                    
                                    # Initialize weather data
                                    weather_data = []
                                    if WEATHER_API_KEY:
                                        try:
                                            weather_fetcher = WeatherDataFetcher(WEATHER_API_KEY)
                                            weather_data = weather_fetcher.fetch_weather_data_in_range(
                                                city=district,
                                                start_date=start_date.strftime('%Y-%m-%d'),
                                                end_date=end_date.strftime('%Y-%m-%d')
                                            )
                                            if weather_data:
                                                st.info("✅ Weather data fetched successfully")
                                            else:
                                                st.warning("⚠️ Could not fetch weather data - using default values")
                                        except Exception as e:
                                            st.warning(f"Weather data fetch failed: {str(e)} - using default values")
                                            weather_data = []
                                    
                                    supply_demand_file_path = "commodity_dummy_dataset_modified.csv"  # Adjust path as needed
                                    supply_demand_loader = None
                                    
                                    try:
                                        supply_demand_loader = SupplyDemandLoader(supply_demand_file_path)
                                        if supply_demand_loader.load_data():
                                            st.info("✅ Supply and demand data loaded successfully")
                                        else:
                                            st.warning("⚠️ Could not load supply and demand data - using default values")
                                            supply_demand_loader = None
                                    except Exception as e:
                                        st.warning(f"Supply and demand data load failed: {str(e)} - using default values")
                                        supply_demand_loader = None
                                    
                                    # Process and merge all data
                                    combined_df = process_and_merge_data(
                                        price_records=commodity_records,
                                        weather_data=weather_data,
                                        supply_demand_loader=supply_demand_loader
                                    )
                                    
                                    if combined_df is not None and not combined_df.empty:                                       
                                        # Convert Arrival_Date back to datetime for display
                                        combined_df['arrival_date'] = pd.to_datetime(combined_df['Arrival_Date'])
                                        
                                        # Rename columns to match display function expectations
                                        display_df = combined_df.rename(columns={
                                            'Min_Price': 'min_price',
                                            'Max_Price': 'max_price',
                                            'Modal_Price': 'modal_price',
                                            'Supply (Metric Tons)': 'supply_metric_tons',
                                            'Demand (Metric Tons)': 'demand_metric_tons',
                                            'CPI': 'consumer_price_index'
                                        })
                                        
                                        # Show data composition
                                        st.info(f"""
                                        **Combined data includes:**
                                        • Commodity prices: {len(commodity_records)} records
                                        • Weather data: {'✅ Available' if weather_data else '❌ Not available'}
                                        • Supply/Demand data: {'✅ Available' if supply_demand_loader and supply_demand_loader.supply_demand_data is not None else '❌ Not available'}
                                        """)
                                        
                                        # Display the analysis
                                        display_price_analysis(display_df, combined=True)
                                        
                                        # Download option for combined data
                                        combined_csv = combined_df.to_csv(index=False)
                                        st.download_button(
                                            "📥 Download Combined Data as CSV", 
                                            data=combined_csv, 
                                            file_name=f"combined_data_{commodity}_{district}_{start_date}_{end_date}.csv", 
                                            mime="text/csv"
                                        )
                                        
                                    else:
                                        st.error("❌ Failed to process combined data. Please check your data sources.")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error processing combined data: {str(e)}")
                                    logger.error(f"Error in combined data processing: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())

    finally:
        try:
            if 'db_handler' in locals() and hasattr(db_handler, 'connection') and db_handler.connection:
                db_handler.connection.close()
                logger.info("Database connection closed")
        except Exception as close_error:
            logger.error(f"Error closing database connection: {str(close_error)}")

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


def render_home_page():
    st.title("🌾 Welcome to KRUSH!")
    

    st.markdown("""
    <div style='padding: 2em; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 2em;'>
        <h3 style='color: #2c3e50; text-align: center;'>
            "Agriculture is our wisest pursuit, because it will in the end contribute most to real wealth, good morals & happiness."
        </h3>
        <p style='text-align: right; color: #7f8c8d;'>- Thomas Jefferson</p>
    </div>
    """, unsafe_allow_html=True)



def main():
    st.set_page_config(
        page_title="KRUSH!",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # st.title("🌾 KRUSH!")
    st.sidebar.title("Navigation")
    pages = {
        "Home": render_home_page,
        "Analytics": analytics_page,
        "Prediction": prediction_page,
    }
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()



if __name__ == "__main__":
    main()


