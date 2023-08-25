import pandas as pd
import numpy as np
from datetime import datetime, date
import calendar

data = pd.read_csv('UserBehavior.csv', encoding='utf')
data.columns = ['user_id', 'item_id', 'category_id', 'behavior', 'time']

print(len(data['user_id'].unique()))
print(len(data['item_id'].unique()))
data = data.drop(columns=['category_id'])
data = data[['user_id', 'item_id', 'time', 'behavior']]
one_hot = pd.get_dummies(data['behavior'])
new_data = data.drop(columns=['behavior'])
new_data = new_data.join(one_hot)
min_timestamp = datetime.timestamp(datetime.strptime('26/11/2017 - 08:00:00', "%d/%m/%Y - %H:%M:%S"))
max_timestamp = datetime.timestamp(datetime.strptime('26/11/2017 - 20:00:00', "%d/%m/%Y - %H:%M:%S"))
min_timestamp = int(min_timestamp)
max_timestamp = int(max_timestamp)
print(min_timestamp)
print(max_timestamp)
final_data = new_data[min_timestamp <= new_data['time']]
final_data = final_data[final_data['time'] <= max_timestamp]
print(len(final_data['user_id'].unique()))
print(len(final_data['item_id'].unique()))
final_data.sort_values("time", inplace=True)
final_data.to_csv('TaobaoLarge.csv', index=False)