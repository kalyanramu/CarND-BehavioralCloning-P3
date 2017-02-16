
import pandas as pd
import numpy as np
df = pd.read_csv('./data/data/driving_log.csv')

balanced = pd.DataFrame()   # Balanced dataset
bins = 20                 # N of bins
max_inbin =80                 # N of examples to include in each bin (at most)

print(df['steering'])
start = 0
for end in np.linspace(0, 1, num=bins):
    print("Start=", start)
    print("End=",end)
    df_range = df[(np.absolute(df.steering) >= start)
                  & (np.absolute(df.steering) < end)]
    range_n = min(df_range.shape[0],max_inbin)
    print("Max bins allowed= ",max_inbin)
    print("Bins found= ",df_range.shape[0])
    #print(range_n)
    if range_n >0:
    	balanced = pd.concat([balanced, df_range.sample(range_n)])
    start = end
balanced.to_csv('./data/data/driving_log_balanced.csv', index=False)
