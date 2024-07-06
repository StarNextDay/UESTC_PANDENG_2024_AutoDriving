import pandas as pd
import os
dirs = os.listdir("data")
if not os.path.exists("preprocessed_data"):
    os.makedirs("preprocessed_data")
for dir in dirs:
    if os.path.isdir("data/" + dir):
        file_dirs = os.listdir("data/" + dir)
        if not os.path.exists("preprocessed_data/" + dir):
            os.makedirs("preprocessed_data/" + dir)        
        for file_dir in file_dirs:
            if file_dir[0:10] != "trajectory":
                file_dir = "data/" + dir + "/" + file_dir
                file_data = pd.read_csv(file_dir)
                file_data.drop(columns=['TOR_cnt'], inplace=True)
                file_data.drop(columns=['time'], inplace=True)
                first_one_index = file_data['isTOR'].idxmax()
                file_data['isTOR'].iloc[first_one_index:] = 1
                col_to_move = 'isTOR'
                file_data[col_to_move] = file_data.pop(col_to_move)
                file_data.to_csv("preprocessed_" + file_dir, index=False)