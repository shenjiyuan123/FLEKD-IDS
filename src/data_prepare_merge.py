import csv
import os


def merge_csv_in_folder(folder, foldername):
    output_file = os.path.join(folder, foldername+'.csv')
    # read all csv files in folder
    csv_files = [os.path.join(folder, foldername, f) for f in os.listdir(os.path.join(folder, foldername)) if f.endswith('.csv')]
    # read header from first csv file
    with open(csv_files[0], 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    # write header to output file if file not exists
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    # write data to output file
    for csv_file in csv_files:
        print("mergeing file: ", csv_file)
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # skip first row
                if row[-1] == 'Label':
                    continue
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
    print("merge csv files in folder {} done".format(os.path.join(folder, foldername)))

# merge_csv_in_folder(r'../CSV-03-11/03-11/split_by_label', 'X_train_Y_train')
# merge_csv_in_folder(r'../CSV-03-11/03-11/split_by_label', 'X_test_Y_test')
# merge_csv_in_folder(r'../CSV-03-11/03-11/split_by_label', 'X_train_79d_Y_train')
# merge_csv_in_folder(r'../CSV-03-11/03-11/split_by_label', 'X_train_24d_Y_train')
merge_csv_in_folder(r'../CSV-03-11/03-11/fl_split_by_label', 'X_test_Y_test')
