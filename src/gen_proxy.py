import pandas as pd
import numpy as np
import random
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)


def main(proxy_num=1260, client_num=9, label_num = 7):
    # random.seed(42)
    each_client_proxy_num = proxy_num // client_num
    each_class_proxy_num = each_client_proxy_num // label_num
    print(each_class_proxy_num)
    csv_file_folder = '/export/home2/jiyuan/CIC2019/CSV-03-11/03-11-V2/fl_split_by_label/X_train_Y_train'
    client_files = [os.path.join(csv_file_folder, "client_" + str(i) + ".csv") for i in range(client_num)]
    out_base_folder = '/export/home2/jiyuan/CIC2019/CSV-03-11/03-11-V2/kd_fl_split_by_label'
    if not os.path.exists(out_base_folder):
        os.mkdir(out_base_folder)
    out_folder = os.path.join(out_base_folder,'X_train_Y_train')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    columns = pd.read_csv(client_files[0]).columns
    proxy_data = pd.DataFrame(columns=columns)
    for i, client_file in enumerate(client_files):
        df = pd.read_csv(client_file)
        out_df = pd.DataFrame(columns=df.columns)
        for label in range(label_num):
            if i%3==0 and label==6:
                df_tmp = df[df['Label']==label]
                df_tmp = df_tmp.reset_index(drop=True)
                out_df = pd.concat([out_df, df_tmp])
            else:
                df_tmp = df[df['Label']==label]
                df_tmp = df_tmp.reset_index(drop=True)
                print(len(df_tmp))
                proxy_indices = random.sample(range(len(df_tmp)), each_class_proxy_num)
                proxy_df = pd.DataFrame(df_tmp.iloc[proxy_indices])
                proxy_data = pd.concat([proxy_data, proxy_df])
                
                df_tmp.drop(index=proxy_indices, inplace=True)
                out_df = pd.concat([out_df, df_tmp])
        
        out_path = os.path.join(out_folder, "client_" + str(i) + ".csv")
        out_df.to_csv(out_path, index=False)
    
    proxy_data = proxy_data.reset_index(drop=True)
    out_path = os.path.join(out_folder,'proxy_data.csv')
    proxy_data.to_csv(out_path, index=False)
    return proxy_data


if __name__=="__main__":
    proxy_data = main()