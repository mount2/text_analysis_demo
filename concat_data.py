import pandas as pd 

df1 = pd.read_csv('concated_data.csv')

df2 = pd.read_csv('data3.csv')

df_tot = pd.concat([df1,df2],ignore_index=True)

df_tot.to_csv('concated_data.csv',index = False)