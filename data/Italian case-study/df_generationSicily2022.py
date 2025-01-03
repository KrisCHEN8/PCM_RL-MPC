
import pandas as pd
from entsoe import EntsoePandasClient


#parameters
start = pd.Timestamp('20220101', tz='Europe/Rome')
end = pd.Timestamp('20230101', tz='Europe/Rome')
country_code = 'IT_SICI'  
range_time_load="15T"
range_time_generation="15T"

#initialise client
client = EntsoePandasClient(api_key='5b6c09de-3782-4898-a937-dcb08794deba')

#make queries to generate the dataframes
df_generation=client.query_generation(country_code, start=start, end=end, psr_type=None)

# Set max columns
pd.set_option('display.max_columns', None)

print(df_generation)