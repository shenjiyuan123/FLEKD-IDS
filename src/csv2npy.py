import csv
import os
import numpy as np
import pandas as pd

def csv2npy(train_csv, test_csv):
    df = pd.read_csv(train_csv)
    # shuffle df
    df = df.sample(frac=1).reset_index(drop=True)
    df_test = pd.read_csv(test_csv)
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    X_train = df.values[:,:-1]
    Y_train = df.values[:,-1]
    df_test = df_test[['ACK Flag Count',
          'Average Packet Size',
          'Destination Port',
          'Flow Duration',
          'Flow IAT Max',
          'Flow IAT Mean',
          'Flow IAT Min',
          'Fwd Header Length',
          'Fwd Header Length.1',
          'Fwd IAT Max',
          'Fwd IAT Mean',
          'Fwd IAT Total',
          'Fwd Packet Length Max',
          'Fwd Packet Length Min',
          'Fwd Packet Length Std',
          'Fwd Packets/s',
          'Init_Win_bytes_forward',
          'Total Length of Fwd Packets',  # 'Length of Fwd Packets',
          'Max Packet Length',
          'Min Packet Length',
          'min_seg_size_forward',
          'Packet Length Std',
          'Protocol',
          'Subflow Fwd Bytes', 'Label']]
    X_test = df_test.values[:,:-1]
    Y_test = df_test.values[:,-1]
    data_set = dict([('X_train', X_train),
                     ('Y_train', Y_train),
                     ('X_test', X_test),
                     ('Y_test', Y_test)])
    np.save('data.npy', data_set)

if __name__ == '__main__':
    csv2npy(r'../CSV-03-11/03-11/split_by_label/X_train_24d_Y_train.csv', r'../CSV-03-11/03-11/split_by_label/X_test_Y_test.csv')