import pandas as pd

def preprocessed(data_path, cols):
    '''
        Returns preprocessed data.
        The proprocessing methods include removing datapoint with bad value or outlier in any of the columns
    '''
    data = pd.read_csv(data_path, usecols=cols)

    # Removing non numeric value
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data.astype('float64')

    # Removing outlier
    q1 = data[cols].quantile(0.25)
    q3 = data[cols].quantile(0.75)
    iqr = q3-q1
    condition = ~((data[cols] < (q1 - 1.5 * iqr)) | (data[cols] > (q3 + 1.5 * iqr))).any(axis=1)
    data = data[condition]

    # Returning pre-processed data
    return data