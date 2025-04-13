import pickle
import pandas as pd
import numpy as np


pkl_file = './S17/S17.pkl'  


with open(pkl_file, 'rb') as file:
    data = pickle.load(file, encoding='latin1')

bvp = data['signal']['wrist']['BVP']  # this one  is at 64 Hz
eda = data['signal']['wrist']['EDA']  # this is  4 Hz
temp = data['signal']['wrist']['TEMP']  #this is at  4 Hz
labels = data['label']  # this is at 700 Hz

# blood volume pressure downsampled to 4hZ from 64Hz 
bvp_downsampled = np.array(bvp[::16]).flatten()

#label downsampled from 700 Hz to 4 Hz
label_downsampled = np.array(labels[::175]).flatten()


min_len = min(len(bvp_downsampled), len(eda), len(temp), len(label_downsampled))


bvp_downsampled = bvp_downsampled[:min_len]
eda = np.array(eda).flatten()[:min_len]
temp = np.array(temp).flatten()[:min_len]
label_downsampled = label_downsampled[:min_len]


combined_df = pd.DataFrame({
    'BVP': bvp_downsampled,
    'EDA': eda,
    'TEMP': temp,
    'Label': label_downsampled
})

combined_df.to_csv("combined_wesad_4Hz.csv", index=False)

print("dataset saved as 'combined_wesad_4Hz.csv'")
