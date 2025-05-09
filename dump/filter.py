import pandas as pd

path='./S17/'
df = pd.read_csv(path+'combined_wesad_4Hz.csv')


print("Original:")
print(df.head())


df_filtered = df[~df['Label'].isin([0, 5, 6, 7])]

# Final filtrered dataset displayu
print("\nFiltered Data after removing labels 0, 5, 6, and 7 which are undefined states of mind:")
print(df_filtered.head())


df_filtered.to_csv(path+'filtered_data.csv', index=False)
