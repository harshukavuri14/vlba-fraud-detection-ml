from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
import numpy as np
import io

def flag_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).astype(int)

def rolling_unique_set(series, window):
    result = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        window_slice = series.iloc[start:i+1]
        result.append(len(set(window_slice)))
    return pd.Series(result, index=series.index)

def load_lookup_csv_gcs(bucket, fname):
    blob = bucket.blob(fname)
    csv_string = blob.download_as_text()
    df = pd.read_csv(io.StringIO(csv_string), index_col=0)
    return df.squeeze("columns")

def feature_engineering_production_bq(
    input_bq_table="vlba-fd.fd.transactions_production",
    output_bq_table="vlba-fd.fd.production_feature_engineered",
    lookup_gcs_bucket="vlba-fd-lookups-bucket"
):
    # --- 1. Read from BigQuery ---
    bq = bigquery.Client()
    print(f"Reading data from: {input_bq_table}")
    df = bq.query(f"SELECT * FROM `{input_bq_table}`").to_dataframe()
    print("Loaded data shape:", df.shape)

    # --- 2. Feature Engineering (all steps) ---
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    df['Log_Amount_Received'] = np.log1p(df['Amount Received'])
    df['Log_Amount_Paid'] = np.log1p(df['Amount Paid'])
    df['Amount_Diff'] = abs(df['Amount Received'] - df['Amount Paid'])
    df['Amount_Ratio'] = df['Amount Received'] / (df['Amount Paid'] + 1)
    df['Outlier_Amount_Received'] = flag_outliers_iqr(df['Amount Received'])
    df['Outlier_Amount_Paid'] = flag_outliers_iqr(df['Amount Paid'])

    sender_txn_counts = df['Account'].value_counts(normalize=False)
    df['sender_total_txn'] = df['Account'].map(sender_txn_counts).fillna(0).astype(int)
    receiver_txn_counts = df['Account.1'].value_counts(normalize=False)
    df['receiver_total_txn'] = df['Account.1'].map(receiver_txn_counts).fillna(0).astype(int)
    unique_receivers = df.groupby('Account')['Account.1'].nunique()
    df['unique_receivers_per_sender'] = df['Account'].map(unique_receivers).fillna(0).astype(int)
    unique_senders = df.groupby('Account.1')['Account'].nunique()
    df['unique_senders_per_receiver'] = df['Account.1'].map(unique_senders).fillna(0).astype(int)
    df['Time_Since_Last_Txn_Sender'] = df.groupby('Account')['Timestamp'].diff().dt.total_seconds()
    df['Time_Since_Last_Txn_Sender'] = df['Time_Since_Last_Txn_Sender'].fillna(-1)
    df['Time_Since_Last_Txn_Receiver'] = df.groupby('Account.1')['Timestamp'].diff().dt.total_seconds()
    df['Time_Since_Last_Txn_Receiver'] = df['Time_Since_Last_Txn_Receiver'].fillna(-1)

    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute

    # Interaction Features
    df['Bank_Pair'] = df['From Bank'].astype(str) + '_' + df['To Bank'].astype(str)
    pair_freq = df['Bank_Pair'].value_counts(normalize=True)
    df['Bank_Pair_freq_enc'] = df['Bank_Pair'].map(pair_freq)
    df['Account_Pair'] = df['Account'].astype(str) + '_' + df['Account.1'].astype(str)
    account_pair_freq = df['Account_Pair'].value_counts(normalize=True)
    df['Account_Pair_freq_enc'] = df['Account_Pair'].map(account_pair_freq)
    df['PaymentFormat_Hour'] = df['Payment Format'].astype(str) + '_' + df['Hour'].astype(str)
    payment_hour_freq = df['PaymentFormat_Hour'].value_counts(normalize=True)
    df['PaymentFormat_Hour_freq_enc'] = df['PaymentFormat_Hour'].map(payment_hour_freq)
    df['Sender_PaymentFormat'] = df['Account'].astype(str) + '_' + df['Payment Format'].astype(str)
    sender_payment_freq = df['Sender_PaymentFormat'].value_counts(normalize=True)
    df['Sender_PaymentFormat_freq_enc'] = df['Sender_PaymentFormat'].map(sender_payment_freq)
    df['Day_Hour'] = df['Day'].astype(str) + '_' + df['Hour'].astype(str)
    day_hour_freq = df['Day_Hour'].value_counts(normalize=True)
    df['Day_Hour_freq_enc'] = df['Day_Hour'].map(day_hour_freq)
    df['Bank_Payment_Hour'] = df['From Bank'].astype(str) + '_' + df['Payment Format'].astype(str) + '_' + df['Hour'].astype(str)
    bank_pay_hour_freq = df['Bank_Payment_Hour'].value_counts(normalize=True)
    df['Bank_Payment_Hour_freq_enc'] = df['Bank_Payment_Hour'].map(bank_pay_hour_freq)

    # Behavioral Features (rolling, velocity, etc.)
    window_size = 5
    df = df.sort_values('Timestamp')

    df['Timestamp_unix'] = df['Timestamp'].astype('int64') // 10**9
    window_seconds = 3600
    def txn_velocity_sender(group):
        times = group['Timestamp_unix'].values
        counts = []
        for i, t in enumerate(times):
            counts.append(((times >= t - window_seconds) & (times <= t)).sum())
        return pd.Series(counts, index=group.index)
    df['Txn_Velocity_Sender'] = df.groupby('Account').apply(txn_velocity_sender).reset_index(level=0, drop=True)
    df.drop(columns=['Timestamp_unix'], inplace=True, errors='ignore')

    df['Rolling_Std_Amount_Received_Sender'] = df.groupby('Account')['Amount Received']\
        .rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    df['Rolling_Std_Amount_Paid_Sender'] = df.groupby('Account')['Amount Paid']\
        .rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    df['Rolling_Std_Amount_Received_Receiver'] = df.groupby('Account.1')['Amount Received']\
        .rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    df['Rolling_Std_Amount_Paid_Receiver'] = df.groupby('Account.1')['Amount Paid']\
        .rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    epsilon = 1e-6
    df['Rolling_Avg_Amount_Received_Sender'] = df.groupby('Account')['Amount Received']\
        .rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Rolling_Avg_Amount_Paid_Sender'] = df.groupby('Account')['Amount Paid']\
        .rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Rolling_Avg_Amount_Received_Receiver'] = df.groupby('Account.1')['Amount Received']\
        .rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Rolling_Avg_Amount_Paid_Receiver'] = df.groupby('Account.1')['Amount Paid']\
        .rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Amount_Received_to_Avg_Sender_Ratio'] = df['Amount Received'] / (df['Rolling_Avg_Amount_Received_Sender'] + epsilon)
    df['Amount_Paid_to_Avg_Sender_Ratio'] = df['Amount Paid'] / (df['Rolling_Avg_Amount_Paid_Sender'] + epsilon)
    df['Amount_Received_to_Avg_Receiver_Ratio'] = df['Amount Received'] / (df['Rolling_Avg_Amount_Received_Receiver'] + epsilon)
    df['Amount_Paid_to_Avg_Receiver_Ratio'] = df['Amount Paid'] / (df['Rolling_Avg_Amount_Paid_Receiver'] + epsilon)

    window_size_txns = 20
    df['Rolling_Unique_Receivers_Sender'] = df.groupby('Account')['Account.1']\
        .apply(lambda x: rolling_unique_set(x, window_size_txns)).reset_index(level=0, drop=True)
    df['Timestamp_unix'] = df['Timestamp'].astype('int64') // 10**9
    def txn_velocity_receiver(group):
        times = group['Timestamp_unix'].values
        counts = []
        for i, t in enumerate(times):
            counts.append(((times >= t - window_seconds) & (times <= t)).sum())
        return pd.Series(counts, index=group.index)
    df['Txn_Velocity_Receiver'] = df.groupby('Account.1').apply(txn_velocity_receiver).reset_index(level=0, drop=True)
    df.drop(columns=['Timestamp_unix'], inplace=True, errors='ignore')

    # Categorical Encoding
    freq_target_cols = ['From Bank', 'To Bank']
    one_hot_cols = ['Receiving Currency', 'Payment Currency', 'Payment Format']
    for col in freq_target_cols:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc)
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
    for col in ['Account', 'Account.1']:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc)

    # --- 3. Load Target-Dependent Features from GCS Lookup Tables ---
    print("Loading lookup tables from GCS...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(lookup_gcs_bucket)

    fraud_rate_by_day = load_lookup_csv_gcs(bucket, 'Fraud_Rate_By_Day_lookup.csv')
    fraud_rate_by_hour = load_lookup_csv_gcs(bucket, 'Fraud_Rate_By_Hour_lookup.csv')
    fraud_rate_by_minute = load_lookup_csv_gcs(bucket, 'Fraud_Rate_By_Minute_lookup.csv')
    sender_fraud_rate = load_lookup_csv_gcs(bucket, 'sender_fraud_rate_lookup.csv')
    receiver_fraud_rate = load_lookup_csv_gcs(bucket, 'receiver_fraud_rate_lookup.csv')
    from_bank_target_enc = load_lookup_csv_gcs(bucket, 'From_Bank_target_enc_lookup.csv')
    to_bank_target_enc = load_lookup_csv_gcs(bucket, 'To_Bank_target_enc_lookup.csv')
    account_target_enc = load_lookup_csv_gcs(bucket, 'Account_target_enc_lookup.csv')
    account1_target_enc = load_lookup_csv_gcs(bucket, 'Account.1_target_enc_lookup.csv')
    day_thresh_series = load_lookup_csv_gcs(bucket, 'day_thresh_lookup.csv')
    hour_thresh_series = load_lookup_csv_gcs(bucket, 'hour_thresh_lookup.csv')
    minute_thresh_series = load_lookup_csv_gcs(bucket, 'minute_thresh_lookup.csv')

    df['Fraud_Rate_By_Day'] = df['Day'].map(fraud_rate_by_day).fillna(0)
    df['Fraud_Rate_By_Hour'] = df['Hour'].map(fraud_rate_by_hour).fillna(0)
    df['Fraud_Rate_By_Minute'] = df['Minute'].map(fraud_rate_by_minute).fillna(0)
    df['sender_fraud_rate'] = df['Account'].map(sender_fraud_rate).fillna(0)
    df['receiver_fraud_rate'] = df['Account.1'].map(receiver_fraud_rate).fillna(0)
    df['From Bank_target_enc'] = df['From Bank'].map(from_bank_target_enc).fillna(0)
    df['To Bank_target_enc'] = df['To Bank'].map(to_bank_target_enc).fillna(0)
    df['Account_target_enc'] = df['Account'].map(account_target_enc).fillna(0)
    df['Account.1_target_enc'] = df['Account.1'].map(account1_target_enc).fillna(0)

    day_thresh = float(day_thresh_series.iloc[0]) if hasattr(day_thresh_series, 'iloc') and not day_thresh_series.empty else 0
    hour_thresh = float(hour_thresh_series.iloc[0]) if hasattr(hour_thresh_series, 'iloc') and not hour_thresh_series.empty else 0
    minute_thresh = float(minute_thresh_series.iloc[0]) if hasattr(minute_thresh_series, 'iloc') and not minute_thresh_series.empty else 0

    df['High_Fraud_Day'] = (df['Fraud_Rate_By_Day'] > day_thresh).astype(int)
    df['High_Fraud_Hour'] = (df['Fraud_Rate_By_Hour'] > hour_thresh).astype(int)
    df['High_Fraud_Minute'] = (df['Fraud_Rate_By_Minute'] > minute_thresh).astype(int)

    # Drop interim columns not used for model
    drop_cols = [
        'Bank_Pair', 'Account_Pair', 'PaymentFormat_Hour', 'Sender_PaymentFormat', 'Day_Hour', 'Bank_Payment_Hour'
    ]
    df_model = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # --- 4. Save to BigQuery ---
    print(f"Writing feature engineered production data to: {output_bq_table}")
    df_model.to_gbq(output_bq_table, project_id="vlba-fd", if_exists="replace")
    print("Done!")

if __name__ == "__main__":
    feature_engineering_production_bq()
