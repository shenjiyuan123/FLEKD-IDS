import os.path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def minmax_scale_values(df, col_name):
    scaler = MinMaxScaler()
    scaler = scaler.fit(df[col_name].values.reshape(-1, 1))
    df_values_standardized = scaler.transform(df[col_name].values.reshape(-1, 1))
    df[col_name] = df_values_standardized


# Helper function for one hot encoding
def encode_text(df, name):
    training_set_dummies = pd.get_dummies(df[name])
    for x in training_set_dummies.columns:
        dummy_name = "{}_{}".format(name, x)
        df[dummy_name] = training_set_dummies[x]
    df.drop(name, axis=1, inplace=True)

def process_single_file(folder, label, total_size, test_size):
    print("processing "+label)
    df = pd.read_csv(os.path.join(folder, label+'.csv'), header=None)
    print("read csv file")

    # Know the features names
    columns = df.iloc[0]
    columns = columns.tolist()
    # Give the features names to the dataframe head
    df.columns = [i.strip() for i in columns]

    # Drop the first row (header)
    df.drop(df.index[0], inplace=True)

    # Remove not relevant and class related columns
    df.drop(columns=["Unnamed: 0", "Flow ID", "Source IP", "Destination IP", "Timestamp"], inplace=True)
    # df = df.drop("NaN", axis=1)

    # Check how many unique labels, monetimes there exist str in the class column, so convert all of them to int first
    # df["category"] = pd.to_numeric(df["category"])
    outcomes = df["Label"].unique()

    df["Label"] = df["Label"].map({'BENIGN': 0, #x
                                     'LDAP': 0,  #1
                                     'MSSQL': 1,  #2
                                     'NetBIOS': 2, #3
                                     'Syn': 3, #4
                                     'UDP': 4, #5
                                     'UDPLag': 5,  #6
                                     'DNS': 7, #x
                                     'NTP': 8, #x
                                     'Portmap': 6,  #6
                                     'WebDDoS': 10, #x
                                     'SNMP': 11, #x
                                     'SSDP': 12, #x
                                     'TFTP': 13}, #x
                                    na_action=None)

    # 使用 to_numeric() 函数来处理第三列，让 pandas 把任意无效输入转为 NaN
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    # 丢弃所有 */s 为 infinity的数据
    df.drop(df[df['Flow Duration'] == 0].index, inplace=True)

    # randomly pick 100000 rows
    df = df.sample(n=total_size, random_state=42)

    sympolic_columns = []  # "flgs_number", "proto_number", "state_number
    label_column = "Label"
    for column in df.columns:
        if column in sympolic_columns:
            encode_text(df, column)
        elif not column == label_column:
            minmax_scale_values(df, column)

    label_column = "Label"
    for column in df.columns:
        if not column == label_column:
            minmax_scale_values(df, column)

    # dataframe to array, remove the index and first row (feature names)
    X, Y = df, df.pop("Label")

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    # todo drop some features to make 79-D dataset and 24-D dataset
    # 79D
    X_train_79d = X_train.drop(columns=["Source Port", "Protocol", "SimillarHTTP",  "Inbound"])
    # 24D
    X_train_24d = X_train[['ACK Flag Count',
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
'Total Length of Fwd Packets', #'Length of Fwd Packets',
'Max Packet Length',
'Min Packet Length',
'min_seg_size_forward',
'Packet Length Std',
'Protocol',
'Subflow Fwd Bytes']]


    # combine X and Y
    X_train_Y_train = pd.concat([X_train, Y_train], axis=1)
    X_test_Y_test = pd.concat([X_test, Y_test], axis=1)
    X_train_79d_Y_train = pd.concat([X_train_79d, Y_train], axis=1)
    X_train_24d_Y_train = pd.concat([X_train_24d, Y_train], axis=1)

    # save to csv
    X_train_Y_train.to_csv(os.path.join(r'../CSV-03-11/03-11/fl_split_by_label/X_train_Y_train', label+'.csv'), index=False)
    X_test_Y_test.to_csv(os.path.join(r'../CSV-03-11/03-11/fl_split_by_label/X_test_Y_test', label+'.csv'), index=False)
    # for FL, don't save 79-d and 24-d here. generate them from X_train_Y_train later
    # X_train_79d_Y_train.to_csv(os.path.join(r'../CSV-03-11/03-11/fl_split_by_label/X_train_79d_Y_train', label+'.csv'), index=False)
    # X_train_24d_Y_train.to_csv(os.path.join(r'../CSV-03-11/03-11/fl_split_by_label/X_train_24d_Y_train', label+'.csv'), index=False)
    print("Done")

# process_single_file(r'V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'LDAP', 100000, 0.1)
# process_single_file(r'V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'MSSQL', 100000, 0.1)
# process_single_file(r'V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'NetBIOS', 100000, 0.1)
# process_single_file(r'V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'Syn', 100000, 0.1)
# process_single_file(r'V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'UDP', 100000, 0.1)
# process_single_file(r'V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'UDPLag', 100000, 0.1)
# process_single_file(r'V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'Portmap', 100000, 0.1)
#### todo ???
# process_single_file('V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'DNS', 100000, 0.1)
# process_single_file('V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'NTP', 100000, 0.1)
# process_single_file('V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'WebDDoS', 439, 100)
# process_single_file('V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'SNMP', 100000, 0.1)
# process_single_file('V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'SSDP', 100000, 0.1)
# process_single_file('V:\GithubDataset\Dataset\CIC2019\CSV-03-11\03-11', 'TFTP', 100000, 0.1)


# process_single_file(r'../CSV-03-11/03-11/split_by_label', 'LDAP', 50000, 0.3)
# process_single_file(r'../CSV-03-11/03-11/split_by_label', 'MSSQL', 50000, 0.3)
# process_single_file(r'../CSV-03-11/03-11/split_by_label', 'NetBIOS', 50000, 0.3)
# process_single_file(r'../CSV-03-11/03-11/split_by_label', 'Syn', 50000, 0.3)
# process_single_file(r'../CSV-03-11/03-11/split_by_label', 'UDP', 50000, 0.3)
# process_single_file(r'../CSV-03-11/03-11/split_by_label', 'UDPLag', 1873, 0.3)
# process_single_file(r'../CSV-03-11/03-11/split_by_label', 'Portmap', 50000, 0.3)
total_size = 350*3+3500*3+35000*3+15000
process_single_file(r'../CSV-03-11/03-11/split_by_label', 'LDAP', total_size, 15000)
process_single_file(r'../CSV-03-11/03-11/split_by_label', 'MSSQL', total_size, 15000)
process_single_file(r'../CSV-03-11/03-11/split_by_label', 'NetBIOS', total_size, 15000)
process_single_file(r'../CSV-03-11/03-11/split_by_label', 'Syn', total_size, 15000)
process_single_file(r'../CSV-03-11/03-11/split_by_label', 'UDP', total_size, 15000)
process_single_file(r'../CSV-03-11/03-11/split_by_label', 'UDPLag', 1873 if total_size > 1873 else total_size, 0.3)
process_single_file(r'../CSV-03-11/03-11/split_by_label', 'Portmap', total_size, 15000)



# save the data
# np.save('X_train.npy', mydataset['X_train'])
# np.save('X_test.npy', mydataset['X_test'])
# np.save('Y_train.npy', mydataset['Y_train'])
# np.save('Y_test.npy', mydataset['Y_test'])
# np.save('X_train_79d.npy', mydataset['X_train_79d'])
# np.save('X_train_24d.npy', mydataset['X_train_24d'])

