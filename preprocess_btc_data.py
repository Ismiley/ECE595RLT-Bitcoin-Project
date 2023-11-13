import pandas as pd
def preprocess_for_training(df):

    # Select only 'Timestamp' and 'Open' columns
    df = df[['Timestamp', 'Open']]

    # Rename the 'Open' column to 'price'
    df = df.rename(columns={'Open': 'Price'})

    # Take the last 10000 entries as a sample, you might want to adjust this
    df = df.head(1000000)

    # Convert UNIX Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

    # Handle NaN values: fill with previous value
    df.ffill(inplace=True)

    # Set the 'Timestamp' column as the index
    df.set_index('Timestamp', inplace=True)

    # Resample to hourly data, taking the mean of the 'Price'
    # You can also use other methods like 'ohlc' if you need open-high-low-close
    df_resampled = df.resample('H').first()

    # Reset index if required
    df_resampled.reset_index(inplace=True)

    return df_resampled


def preprocess_for_testing(df):

    # Select only 'Timestamp' and 'Open' columns
    df = df[['Timestamp', 'Open']]

    # Rename the 'Open' column to 'price'
    df = df.rename(columns={'Open': 'Price'})

    # Take the last 10000 entries as a sample, you might want to adjust this
    df = df.tail(1000000)

    # Convert UNIX Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

    # Handle NaN values: fill with previous value
    df.ffill(inplace=True)

    # Set the 'Timestamp' column as the index
    df.set_index('Timestamp', inplace=True)

    # Resample to hourly data, taking the mean of the 'Price'
    # You can also use other methods like 'ohlc' if you need open-high-low-close
    df_resampled = df.resample('H').first()

    # Reset index if required
    df_resampled.reset_index(inplace=True)

    return df_resampled