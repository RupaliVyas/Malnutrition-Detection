import pandas as pd
df = pd.read_csv('dataset.csv')
# for i in range(4):
#     for j in range(len(df.index)):
#         print(df.iloc[j,i])
# print(df.columns)
# df['class']='Nan'


df['diff']=abs(df['weight'] - df['Mean weight'])
# print(df.head(4))

df['class']=[0 if x<2.0 else 2 for x in df['diff']]
# df['class']=['moderate malnutrtion' if 0.9<x<2.0 else print("") for x in df['diff']]

df2 = df.loc[df['class'] == 0]
df2 = df2.reset_index()
df2['class']=[0 if x<1.2 else 1 for x in df2['diff']]

# print(df2.head(10))
#df2.to_csv('data.csv', index=False)
# for index,row in df.iterrows():
#     if row['class']==0:
#         df.drop(row)
#df = df.drop[df.loc[df['class'] == 0]]
df = df[df['class'] !=0]
df = df.reset_index()
# print(df.head(30))

df = df.append(df2)

print(df.head(1000))

df.to_csv('final_dataset.csv', index=False)

