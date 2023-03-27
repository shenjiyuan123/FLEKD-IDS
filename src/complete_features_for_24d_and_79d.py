import csv

import pandas as pd

csv_total_path = r'../CSV-03-11/03-11/split_by_label/X_test_Y_test.csv'
header_total = []
# read header
with open(csv_total_path, 'r') as f:
    reader = csv.reader(f)
    header_total = next(reader)

def complete_features(csv_path, header_total):
    df = pd.read_csv(csv_path)
    # get header from df
    header = list(df.columns)
    # get missing header from header_total
    missing_header = [h for h in header_total if h not in header]
    # add missing header to df
    for h in missing_header:
        df[h] = 0
    # move label to last column
    df = df[header_total]

    # write to csv
    df.to_csv(csv_path, index=False)


csv_79d_path = r'../CSV-03-11/03-11/split_by_label/X_train_79d_Y_train.csv'

complete_features(csv_79d_path, header_total)

csv_24d_path = r'../CSV-03-11/03-11/split_by_label/X_train_24d_Y_train.csv'

complete_features(csv_24d_path, header_total)