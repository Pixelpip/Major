import pandas as pd


path='./S17/'
df = pd.read_csv(path+'filtered_data.csv')

k = 1.5 #found to be the most commonly used constant for IQR 

for column in df.select_dtypes(include=['float64', 'int64']).columns:
    
    Q1 = df[column].quantile(0.25) 
    Q3 = df[column].quantile(0.75)
    

    IQR = Q3 - Q1
    
    # Upper and Lower Fences formula obtained from the research paper
    UF = Q3 + k * IQR
    LF = Q1 - k * IQR
    
    # Remove rows which are outside the fence
    df = df[(df[column] >= LF) & (df[column] <= UF)]


print(df.head())
df.to_csv(path+'IQR.csv',index=False)
