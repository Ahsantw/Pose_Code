import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_name',required=True)
args = parser.parse_args() 

df1=pd.read_csv(args.csv_name)


df = df1[:-int(len(df1)*0.15)]
df_val = df1[-int(len(df1)*0.15):]

print(len(df))
print(len(df_val))
df.to_csv('instances_train2017.csv',index=False)
df_val.to_csv('instances_val2017.csv',index=False)