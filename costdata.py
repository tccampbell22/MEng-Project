#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 11:03:23 2024

@author: thomascampbell
"""
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np


def capital_cost_dataframe(start):
    """
    interpolates the cost csv to get capex for each year, then picks out
    data for the simulation year
    """
    
    start_ = int(start[0:4])
    # Replace 'your_file.csv' with the actual path to your CSV file
    file_path = '../data/Capex costs.xlsx'

    # Read the CSV file into a DataFrame
    capex = pd.read_excel(file_path, sheet_name="Capex costs", index_col=0)
    #resize the dataframe to include each year within the date range
    new_index = range(2010, 2051)
    capex=capex.reindex(new_index, axis=1)
    
    #extrapolate to fill in missing years for each generator type
    Generators = capex.index
    
    for generator in Generators:
        data=capex.loc[generator]
        data = data.dropna()
        x = data.index
        y = data.values
        f = interp1d(x, y, fill_value='extrapolate')
        interpolated_values = f(new_index)
        interpolated_data = pd.Series(interpolated_values, index=new_index)
        capex.loc[generator]=interpolated_data
        
    capex=capex.T
    df=0.86*capex.loc[[2010,2015,2020,2025,2030,2035,2040,2045,2050]]
    for gen in Generators:
        print(df[gen])
    capex=capex.loc[start_]
    
    
    return capex
    



def write_capital_costs_series(start, end, freq, year):
    
    #adds a data column capital_cost to generators and stores csvs
    # same method of adding cost data as marginal_costs program

    df_gens = pd.read_csv('LOPF_data/generators.csv',index_col=0)   
    capital_cost_df = capital_cost_dataframe(start)

    capital_costs=np.empty(0)
    
    #gets the cost data in the order of the generators, then adds a cap cost column to
    #generators.csv file. Does this for gens and stores
    for gen in df_gens.index:
        capital_costs=np.append(capital_costs, capital_cost_df[df_gens.loc[gen].type])
            
    
    df_gens['capital_cost']=capital_costs*0.86
    #capex costs in euros so converting to £
    
    df_gens.to_csv('LOPF_data/generators.csv', header=True)
    
    
    
    df_stores = pd.read_csv('LOPF_data/storage_units.csv',index_col=0)   

    capital_costs=np.empty(0)
    
    for store in df_stores.index:
        capital_costs=np.append(capital_costs, capital_cost_df[df_stores.loc[store].carrier])
            
    
    df_stores['capital_cost']=capital_costs*0.86
    #capex costs in euros so converting to £
    
    
    df_stores.to_csv('LOPF_data/storage_units.csv', header=True)


