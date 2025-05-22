"""
This script is used to concatenate the data from the different files into a single file.
"""

import pandas as pd
import os
import argparse


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Concatenate SIATA station data files."
    )
    parser.add_argument(
        "--station",
        type=str,
        required=True,
        help='Station number to process (e.g., "28")',
    )

    # Parse arguments
    args = parser.parse_args()
    station = args.station

    # Get the list of files in the folder
    folder_path = f"data/SIATA/{station}"
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    files = os.listdir(folder_path)

    # Create an empty list to store the data
    data = []

    # Loop through the files
    for file in files:
        try:
            # Read the data from the file
            df = pd.read_csv(f"{folder_path}/{file}")

            # Detect unnamed columns
            unnamed_columns = [col for col in df.columns if "Unnamed" in col]

            if unnamed_columns:
                print(
                    f"Unnamed columns detected in {file} in the positions {unnamed_columns}"
                )

            if "Unnamed: 0" in df.columns:
                # Replace the name of the first column
                df.rename(columns={"Unnamed: 0": "Fecha_Hora"}, inplace=True)

            # Append the data to the list
            data.append(df)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")

    # Concatenate the data
    data = pd.concat(data)

    # Save the data to a file
    output_file = f"data/SIATA/{station}.csv"
    data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    main()
