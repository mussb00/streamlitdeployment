import pandas as pd
from datetime import datetime

def main():
    url = 'https://hidmet.gov.rs/eng/hidrologija/godisnje/godisnjak.php?sifra=45099'
    dfs = pd.read_html(url)
    #The dataframe of interest is the first one.
    df = dfs[0]

    df = df.iloc[:-2] #Removing the last 2 rows, because they don't contain data of interest
    df.columns = df.columns.map(''.join) #The original columns are multi indexed. This converts them in simple columns
    df.columns = ['day', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] #Renaming the columns properly.

    #Transforming the dataframe in a format which 2 columns correspond to days
    #and months, and the third column contains the values.
    df_melted = pd.melt(df, id_vars=['day'],
                        var_name='month',
                        value_name='belgrade_water_level_cm')

    cur_year = datetime.now().year
    #Creating a column of dates.
    df_melted['date'] = pd.to_datetime(
        df_melted['day'].astype(str) + '-' + df_melted['month'].astype(str) + '-' + str(cur_year),
        format='%d-%m-%Y',
        errors='coerce')

    #Removing rows containing no dates (since there were rows containing day 30 and
    #month 2 (feb), for example, this implied in empty date).
    df_melted = df_melted.dropna(subset=['date'])

    #Removing columns Day and Month. Keeping only date and water levels.
    df_melted = df_melted[['date', 'belgrade_water_level_cm']]
    df_melted = df_melted.dropna(subset=['belgrade_water_level_cm'])
    df_melted['date'] = [d.date() for d in df_melted['date']]
    df_melted['belgrade_water_level_cm'] = df_melted['belgrade_water_level_cm'].astype(float)
    return df_melted
