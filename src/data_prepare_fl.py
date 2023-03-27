import csv
import os
import pandas as pd

def convert_to_79d_with_zero_filling(df):
    df_79d = df.copy()
    columns=["Source Port", "Protocol", "SimillarHTTP",  "Inbound"]
    for i in columns:
        df_79d[i] = 0
    return df_79d


def convert_to_24d_with_zero_filling(df):
    df_24d = df.copy()
    columns=['ACK Flag Count',
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
'Subflow Fwd Bytes']
    for i in df_24d.columns:
        if i not in columns:
            df_24d[i] = 0
    return df_24d


def write_output_file(output_file, df):
    # write df to csv, append if file already exists
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)


CLIENTS = ['0:350:full', '1:3500:full', '2:35000:full',
           '3:350:79d', '4:3500:79d', '5:35000:79d',
           '6:350:24d', '7:3500:24d', '8:35000:24d']

def merge_csv_in_folder(folder, foldername):
    # read all csv files in folder
    csv_files = [os.path.join(folder, foldername, f) for f in os.listdir(os.path.join(folder, foldername)) if f.endswith('.csv')]
    for csv_file in csv_files:
        print("processing file: ", csv_file)
        # WARNING: assume column names are the same in all csv files, and they should not be changed in the following code
        df = pd.read_csv(csv_file)
        i = 0
        for client in CLIENTS:
            print("processing client: ", client)
            client_id, num, type = client.split(':')

            if 'UDPLag' in csv_file:
                # num cannot exceed the number of rows in the csv file
                if int(num) > len(df):
                    # use all rows in the csv file in case of not enough rows
                    df_client = df.copy()
                else:
                    # randomly select num rows from the csv file
                    df_client = df.sample(n=int(num)) #, random_state=0
            else:
                # pop first num from df
                df_client = df.iloc[i:i+int(num)]
                i += int(num)

            if type == 'full':
                pass
            elif type == '79d':
                df_client = convert_to_79d_with_zero_filling(df_client)
            elif type == '24d':
                df_client = convert_to_24d_with_zero_filling(df_client)

            # write to csv
            output_file = os.path.join(folder, foldername, 'client_' + str(client_id) + '.csv')
            write_output_file(output_file, df_client)


    print("merge csv files in folder {} done".format(os.path.join(folder, foldername)))

merge_csv_in_folder('../CSV-03-11/03-11/fl_split_by_label', 'X_train_Y_train')