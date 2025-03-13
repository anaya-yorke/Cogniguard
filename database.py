import json
from datetime import datetime
import pandas as pd

# Placeholder for actual Snowflake connector
# In a real implementation, you would use:
# import snowflake.connector

class SnowflakeConnection:
    def __init__(self, user=None, password=None, account=None, warehouse=None, database=None, schema=None):
        self.user = user or 'YOUR_SNOWFLAKE_USER'
        self.password = password or 'YOUR_SNOWFLAKE_PASSWORD'
        self.account = account or 'YOUR_SNOWFLAKE_ACCOUNT'
        self.warehouse = warehouse or 'YOUR_WAREHOUSE'
        self.database = database or 'COGNIGUARD_DB'
        self.schema = schema or 'PUBLIC'
        self.measurements = []  # In-memory storage for demo when Snowflake isn't available

    def cursor(self):
        return self

    def execute(self, query, params=None):
        # Demo implementation for execute
        self.last_query = query
        self.last_params = params
        
        # For select queries, return data from in-memory storage
        if query.strip().upper().startswith("SELECT"):
            self.results = self._process_select_query(query, params)
            self.description = self._get_description_for_query(query)
        
        return self
    
    def fetchall(self):
        # Return results for select queries
        return self.results if hasattr(self, 'results') else []
    
    def _process_select_query(self, query, params):
        # Simple query processor for demo purposes
        if not self.measurements:
            return []
            
        # Convert in-memory data to list of tuples for SELECT * queries
        if "cognitive_measurements" in query and "*" in query:
            results = []
            for m in self.measurements:
                result = (
                    m.get('measurement_id', ''),
                    m.get('timestamp', datetime.now()),
                    m.get('overload_probability', 0.0),
                    m.get('is_overloaded', False),
                    m.get('alert_message', ''),
                    m.get('eeg_features', '{}')
                )
                results.append(result)
            
            # Apply limit if present
            if params and len(params) > 0 and "LIMIT" in query:
                limit = params[-1]
                return results[:limit]
            
            return results
            
        # Process aggregated statistics queries
        if "DATE_TRUNC" in query and "GROUP BY" in query:
            # Create sample statistics
            if not self.measurements:
                return []
                
            dates = sorted(set(m['timestamp'].date() for m in self.measurements))
            stats = []
            
            for date in dates:
                day_measurements = [m for m in self.measurements if m['timestamp'].date() == date]
                overload_count = sum(1 for m in day_measurements if m['is_overloaded'])
                avg_prob = sum(m['overload_probability'] for m in day_measurements) / len(day_measurements)
                
                stats.append((date, len(day_measurements), overload_count, avg_prob))
            
            return stats
                
        return []
    
    def _get_description_for_query(self, query):
        # Return column descriptions based on query type
        if "cognitive_measurements" in query and "*" in query:
            return [
                ('MEASUREMENT_ID', None, None, None, None, None, None),
                ('TIMESTAMP', None, None, None, None, None, None),
                ('OVERLOAD_PROBABILITY', None, None, None, None, None, None),
                ('IS_OVERLOADED', None, None, None, None, None, None),
                ('ALERT_MESSAGE', None, None, None, None, None, None),
                ('EEG_FEATURES', None, None, None, None, None, None)
            ]
        elif "DATE_TRUNC" in query and "GROUP BY" in query:
            return [
                ('DAY', None, None, None, None, None, None),
                ('TOTAL_MEASUREMENTS', None, None, None, None, None, None),
                ('OVERLOAD_COUNT', None, None, None, None, None, None),
                ('AVG_PROBABILITY', None, None, None, None, None, None)
            ]
        return []
    
    def commit(self):
        # In a real implementation this would commit the transaction
        pass
    
    def close(self):
        # In a real implementation this would close the connection
        pass

def initialize_snowflake_connection(user=None, password=None, account=None, warehouse=None, database=None, schema=None):
    """
    Initialize connection to Snowflake.
    For demo purposes, this returns a mock connection when credentials aren't provided.
    """
    try:
        # If we have snowflake.connector and valid credentials, use the real connector
        if all([user, password, account, warehouse, database, schema]):
            import snowflake.connector
            conn = snowflake.connector.connect(
                user=user,
                password=password,
                account=account,
                warehouse=warehouse,
                database=database,
                schema=schema
            )
            
            cursor = conn.cursor()
            
            create_tables_query = """
            CREATE TABLE IF NOT EXISTS cognitive_measurements (
                measurement_id VARCHAR(36) DEFAULT UUID_STRING(),
                timestamp TIMESTAMP_NTZ,
                overload_probability FLOAT,
                is_overloaded BOOLEAN,
                alert_message VARCHAR,
                eeg_features VARIANT,
                PRIMARY KEY (measurement_id)
            )
            """
            
            cursor.execute(create_tables_query)
            conn.commit()
            
            return conn
        else:
            # Use our mock implementation
            conn = SnowflakeConnection(user, password, account, warehouse, database, schema)
            return conn
    except ImportError:
        # If snowflake.connector is not available, use our mock implementation
        print("Snowflake connector not available. Using mock implementation for demo.")
        conn = SnowflakeConnection(user, password, account, warehouse, database, schema)
        return conn
    except Exception as e:
        print(f"Error connecting to Snowflake: {str(e)}")
        print("Using mock implementation for demo.")
        conn = SnowflakeConnection(user, password, account, warehouse, database, schema)
        return conn

def store_measurement(conn, prediction, eeg_features=None, alert_message=None):
    """Store a measurement in Snowflake or in-memory store"""
    cursor = conn.cursor()
    
    timestamp = datetime.fromtimestamp(prediction['timestamp'])
    
    features_json = json.dumps(eeg_features.tolist() if hasattr(eeg_features, 'tolist') else eeg_features) if eeg_features else None
    
    try:
        # If this is a real Snowflake connection
        if hasattr(conn, 'measurements'):
            # This is our mock connector
            import uuid
            measurement = {
                'measurement_id': str(uuid.uuid4()),
                'timestamp': timestamp,
                'overload_probability': prediction['overload_probability'],
                'is_overloaded': prediction['is_overloaded'],
                'alert_message': alert_message,
                'eeg_features': features_json
            }
            conn.measurements.append(measurement)
        else:
            # Real Snowflake connection
            insert_query = """
            INSERT INTO cognitive_measurements (
                timestamp, 
                overload_probability, 
                is_overloaded, 
                alert_message,
                eeg_features
            ) VALUES (%s, %s, %s, %s, %s)
            """
            
            cursor.execute(
                insert_query, 
                (timestamp, prediction['overload_probability'], prediction['is_overloaded'], alert_message, features_json)
            )
            
        conn.commit()
    except Exception as e:
        print(f"Error storing measurement: {str(e)}")

def get_historical_measurements(conn, start_time=None, end_time=None, limit=100):
    """Get historical measurements from Snowflake or in-memory store"""
    cursor = conn.cursor()
    
    query = "SELECT * FROM cognitive_measurements"
    
    conditions = []
    params = []
    
    if start_time:
        conditions.append("timestamp >= %s")
        params.append(start_time)
    
    if end_time:
        conditions.append("timestamp <= %s")
        params.append(end_time)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY timestamp DESC LIMIT %s"
    params.append(limit)
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(results, columns=columns)
    
    return df

def get_overload_statistics(conn, period_days=30):
    """Get statistics about overload events from Snowflake or in-memory store"""
    cursor = conn.cursor()
    
    query = """
    SELECT 
        DATE_TRUNC('day', timestamp) as day,
        COUNT(*) as total_measurements,
        SUM(CASE WHEN is_overloaded THEN 1 ELSE 0 END) as overload_count,
        AVG(overload_probability) as avg_probability
    FROM cognitive_measurements
    WHERE timestamp >= DATEADD(day, -%s, CURRENT_TIMESTAMP())
    GROUP BY DATE_TRUNC('day', timestamp)
    ORDER BY day
    """
    
    cursor.execute(query, (period_days,))
    results = cursor.fetchall()
    
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(results, columns=columns)
    
    return df 