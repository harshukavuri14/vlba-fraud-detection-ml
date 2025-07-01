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

def feature_engineering_train_bq(
    input_bq_table="vlba-fd.fd.transactions_train",
    output_bq_table="vlba-fd.fd.feature_engineered",
    lookup_gcs_bucket="vlba-fd-lookups-bucket"
):
    # --- 1. Read from BigQuery ---
    bq = bigquery.Client()
    print(f"Reading data from: {input_bq_table}")
    df = bq.query(f"SELECT * FROM `{input_bq_table}`").to_dataframe()
    print("Loaded data shape:", df.shape)

    # --- 2. Feature Engineering (all logic included) ---
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    # Transaction Amount Features
    df['Log_Amount_Received'] = np.log1p(df['Amount Received'])
    df['Log_Amount_Paid'] = np.log1p(df['Amount Paid'])
    df['Amount_Diff'] = abs(df['Amount Received'] - df['Amount Paid'])
    df['Amount_Ratio'] = df['Amount Received'] / (df['Amount Paid'] + 1)
    df['Outlier_Amount_Received'] = flag_outliers_iqr(df['Amount Received'])
    df['Outlier_Amount_Paid'] = flag_outliers_iqr(df['Amount Paid'])

    # Account-Based Features
    sender_agg = df.groupby('Account').agg(
        sender_total_txn=('Is Laundering', 'count'),
        sender_fraud_txn=('Is Laundering', 'sum')
    ).reset_index()
    sender_agg['sender_fraud_rate'] = sender_agg['sender_fraud_txn'] / sender_agg['sender_total_txn']

    receiver_agg = df.groupby('Account.1').agg(
        receiver_total_txn=('Is Laundering', 'count'),
        receiver_fraud_txn=('Is Laundering', 'sum')
    ).reset_index()
    receiver_agg['receiver_fraud_rate'] = receiver_agg['receiver_fraud_txn'] / receiver_agg['receiver_total_txn']

    df = df.merge(sender_agg[['Account', 'sender_total_txn', 'sender_fraud_rate']], on='Account', how='left')
    df = df.merge(receiver_agg[['Account.1', 'receiver_total_txn', 'receiver_fraud_rate']], on='Account.1', how='left')

    unique_receivers = df.groupby('Account')['Account.1'].nunique().reset_index().rename(columns={'Account.1':'unique_receivers_per_sender'})
    df = df.merge(unique_receivers, on='Account', how='left')
    unique_senders = df.groupby('Account.1')['Account'].nunique().reset_index().rename(columns={'Account':'unique_senders_per_receiver'})
    df = df.merge(unique_senders, on='Account.1', how='left')

    df['Time_Since_Last_Txn_Sender'] = df.groupby('Account')['Timestamp'].diff().dt.total_seconds()
    df['Time_Since_Last_Txn_Sender'] = df['Time_Since_Last_Txn_Sender'].fillna(-1)
    df['Time_Since_Last_Txn_Receiver'] = df.groupby('Account.1')['Timestamp'].diff().dt.total_seconds()
    df['Time_Since_Last_Txn_Receiver'] = df['Time_Since_Last_Txn_Receiver'].fillna(-1)

    # Time-based features
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute

    fraud_rate_by_day = df.groupby('Day')['Is Laundering'].mean()
    fraud_rate_by_hour = df.groupby('Hour')['Is Laundering'].mean()
    fraud_rate_by_minute = df.groupby('Minute')['Is Laundering'].mean()

    df['Fraud_Rate_By_Day'] = df['Day'].map(fraud_rate_by_day)
    df['Fraud_Rate_By_Hour'] = df['Hour'].map(fraud_rate_by_hour)
    df['Fraud_Rate_By_Minute'] = df['Minute'].map(fraud_rate_by_minute)

    day_thresh = fraud_rate_by_day.mean()
    hour_thresh = fraud_rate_by_hour.mean()
    minute_thresh = fraud_rate_by_minute.mean()
    df['High_Fraud_Day'] = (df['Fraud_Rate_By_Day'] > day_thresh).astype(int)
    df['High_Fraud_Hour'] = (df['Fraud_Rate_By_Hour'] > hour_thresh).astype(int)
    df['High_Fraud_Minute'] = (df['Fraud_Rate_By_Minute'] > minute_thresh).astype(int)

    # Interaction Features (pairs and frequencies)
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

    # Behavioral Features (rolling, std, avg, velocity, rolling unique)
    window_size = 5
    df = df.sort_values('Timestamp') # ensure sorted

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
    df['Rolling_Unique_Receivers_Sender'] = df.groupby('Account')['Account.1'].apply(lambda x: rolling_unique_set(x, window_size_txns)).reset_index(level=0, drop=True)

    df['Timestamp_unix'] = df['Timestamp'].astype('int64') // 10**9
    def txn_velocity_receiver(group):
        times = group['Timestamp_unix'].values
        counts = []
        for i, t in enumerate(times):
            counts.append(((times >= t - window_seconds) & (times <= t)).sum())
        return pd.Series(counts, index=group.index)
    df['Txn_Velocity_Receiver'] = df.groupby('Account.1').apply(txn_velocity_receiver).reset_index(level=0, drop=True)
    df.drop(columns=['Timestamp_unix'], inplace=True, errors='ignore')

    # Categorical Encoding (One-hot, freq, target)
    freq_target_cols = ['From Bank', 'To Bank']
    one_hot_cols = ['Receiving Currency', 'Payment Currency', 'Payment Format']
    for col in freq_target_cols:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc)
    for col in freq_target_cols:
        target_enc = df.groupby(col)['Is Laundering'].mean()
        df[col + '_target_enc'] = df[col].map(target_enc)
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
    for col in ['Account', 'Account.1']:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc)
        target_enc = df.groupby(col)['Is Laundering'].mean()
        df[col + '_target_enc'] = df[col].map(target_enc)

    # Drop unnecessary/interim columns
    drop_cols = [
        'Bank_Pair', 'Account_Pair', 'PaymentFormat_Hour', 'Sender_PaymentFormat',
        'Day_Hour', 'Bank_Payment_Hour', 'Amount Received', 'Amount Paid'
    ]
    df_model = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # --- 3. Save feature engineered table to BigQuery ---
    print(f"Writing feature engineered data to: {output_bq_table}")
    df_model.to_gbq(output_bq_table, project_id="vlba-fd", if_exists="replace")

    # --- 4. Upload Lookup Tables to GCS ---
    print("Uploading lookup tables to GCS...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(lookup_gcs_bucket)
    def save_lookup(series, fname):
        csv_buf = io.StringIO()
        series.to_csv(csv_buf)
        blob = bucket.blob(fname)
        blob.upload_from_string(csv_buf.getvalue(), content_type='text/csv')
        print("Uploaded", fname)
    save_lookup(fraud_rate_by_day, 'Fraud_Rate_By_Day_lookup.csv')
    save_lookup(fraud_rate_by_hour, 'Fraud_Rate_By_Hour_lookup.csv')
    save_lookup(fraud_rate_by_minute, 'Fraud_Rate_By_Minute_lookup.csv')
    save_lookup(df_model.groupby('Account')['sender_fraud_rate'].first(), 'sender_fraud_rate_lookup.csv')
    save_lookup(df_model.groupby('Account.1')['receiver_fraud_rate'].first(), 'receiver_fraud_rate_lookup.csv')
    save_lookup(df_model.groupby('From Bank')['From Bank_target_enc'].first(), 'From_Bank_target_enc_lookup.csv')
    save_lookup(df_model.groupby('To Bank')['To Bank_target_enc'].first(), 'To_Bank_target_enc_lookup.csv')
    save_lookup(df_model.groupby('Account')['Account_target_enc'].first(), 'Account_target_enc_lookup.csv')
    save_lookup(df_model.groupby('Account.1')['Account.1_target_enc'].first(), 'Account.1_target_enc_lookup.csv')
    save_lookup(pd.Series({'day_thresh': day_thresh}), 'day_thresh_lookup.csv')
    save_lookup(pd.Series({'hour_thresh': hour_thresh}), 'hour_thresh_lookup.csv')
    save_lookup(pd.Series({'minute_thresh': minute_thresh}), 'minute_thresh_lookup.csv')
    print("All lookup tables uploaded to GCS.")

if __name__ == "__main__":
    feature_engineering_train_bq()
