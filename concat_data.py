'''
    This script is used to concatenate the data from the different files into a single file.
'''


import pandas as pd
import os

# Read the data from the data/SIATA/12 folder

# Get the list of files in the folder
files = os.listdir('data/SIATA/12')

# Create an empty list to store the data
data = []

# Loop through the files
for file in files:
    # Read the data from the file
    df = pd.read_csv(f'data/SIATA/12/{file}')
    
    # Append the data to the list
    data.append(df)

# Concatenate the data
data = pd.concat(data)

# Save the data to a file

data.to_csv('data/SIATA/12.csv', index=False)
