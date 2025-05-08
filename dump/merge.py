import pandas as pd

# Read the CSV files
df1 = pd.read_csv('/home/pc480/Downloads/S2_60sec_statistical_features(1).csv')
df2 = pd.read_csv('/home/pc480/Downloads/S3_60sec_statistical_features.csv')
df3 = pd.read_csv('/home/pc480/Downloads/S4_60sec_statistical_features.csv')
df4 = pd.read_csv('/home/pc480/Downloads/S5_60sec_statistical_features.csv')
df5 = pd.read_csv('/home/pc480/Downloads/S6_60sec_statistical_features.csv')
df6 = pd.read_csv('/home/pc480/Downloads/S7_60sec_statistical_features.csv')
df7 = pd.read_csv('/home/pc480/Downloads/S8_60sec_statistical_features.csv')
df8 = pd.read_csv('/home/pc480/Downloads/S9_60sec_statistical_features.csv')
df9 = pd.read_csv('/home/pc480/Downloads/S10_60sec_statistical_features.csv')
df10 = pd.read_csv('/home/pc480/Downloads/S11_60sec_statistical_features.csv')
df11 = pd.read_csv('/home/pc480/Downloads/S14_60sec_statistical_features.csv')
df12 = pd.read_csv('/home/pc480/Downloads/S17_60sec_statistical_features.csv')
# Concatenate the data (ignore the header in the second file)
merged_df = pd.concat([df1, df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12], ignore_index=True)

# Write the merged data to a new CSV file
merged_df.to_csv('merged_file.csv', index=False)
