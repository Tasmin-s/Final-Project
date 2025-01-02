import pandas as pd
import os

def load_data(file_path):
    """
    Load raw data from a specified file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)

def clean_data(df, columns_to_check):
    """
    Clean the loaded data:
    - Remove rows with missing values in specific columns.
    - Remove duplicates.
    - Standardize column names.
    """
    # Normalize column names: lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Print column names after normalization for better debugging
    print("Normalized Columns in DataFrame:")
    print(df.columns.tolist())  # Print actual normalized columns
    
    # Ensure the columns_to_check list matches the actual DataFrame columns
    for col in columns_to_check:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame columns.")
    
    # Remove rows with missing values in the specified columns
    df.dropna(subset=columns_to_check, inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned data to the specified output path.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

def merge_data(infant_mortality_df, life_expectancy_df):
    """
    Merge the infant mortality and life expectancy datasets on 'year' and 'entity'.
    """
    # Merge the two datasets on 'year' and 'entity'
    merged_df = pd.merge(infant_mortality_df, life_expectancy_df, on=['year', 'entity'], how='inner')
    
    return merged_df

def aggregated_values(merged_df):
    """
    Function to compute aggregated values for infant mortality and life expectancy
    based on both male and female data and add them as new columns in the DataFrame.
    """
    merged_df['aggregated_infant_mortality'] = (
        merged_df['observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'] +
        merged_df['observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births']
    ) / 2

    merged_df['aggregated_life_expectancy'] = (
        merged_df['period_life_expectancy_-_sex:_female_-_age:_0'] +
        merged_df['period_life_expectancy_-_sex:_male_-_age:_0']
    ) / 2

    return merged_df 


def process_data():
    """
    Main function to load, clean, merge, and save data for both datasets.
    """
    # Define paths for the raw data files
    infant_mortality_raw_path = '/Users/Tasmin/Final-Project/data/raw/infant-mortality-rate-wdi.csv'  
    life_expectancy_raw_path = '/Users/Tasmin/Final-Project/data/raw/life-expectation-at-birth-by-sex.csv'  
    
    # Define paths for the output (cleaned) data files
    infant_mortality_cleaned_path = '/Users/Tasmin/Final-Project/data/processed/infant-mortality-rate-wdi-cleaned.csv'
    life_expectancy_cleaned_path = '/Users/Tasmin/Final-Project/data/processed/life-expectation-at-birth-by-sex-cleaned.csv'  
    merged_data_path = '/Users/Tasmin/Final-Project/data/processed/merged_data.csv'
    
    # Columns to check for missing values in infant mortality data
    infant_mortality_columns = [
        'entity',
        'year',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
    ]
    
    # Columns to check for missing values in life expectancy data
    life_expectancy_columns = [
        'entity',
        'year',
        'period_life_expectancy_-_sex:_female_-_age:_0',
        'period_life_expectancy_-_sex:_male_-_age:_0'
    ]
    
    # Load, clean, and save data for infant mortality
    print("Processing Infant Mortality Data...")
    infant_mortality_df = load_data(infant_mortality_raw_path)
    cleaned_infant_mortality_df = clean_data(infant_mortality_df, infant_mortality_columns)
    save_cleaned_data(cleaned_infant_mortality_df, infant_mortality_cleaned_path)
    
    # Load, clean, and save data for life expectancy
    print("Processing Life Expectancy Data...")
    life_expectancy_df = load_data(life_expectancy_raw_path)
    cleaned_life_expectancy_df = clean_data(life_expectancy_df, life_expectancy_columns)
    save_cleaned_data(cleaned_life_expectancy_df, life_expectancy_cleaned_path)
    
    # Merge the cleaned data on 'year' and 'entity'
    print("Merging Data...")
    merged_df = merge_data(cleaned_infant_mortality_df, cleaned_life_expectancy_df)
    
    # Apply aggregation
    print("Applying Aggregation...")
    merged_df = aggregated_values(merged_df)
    
    # Save merged data
    save_cleaned_data(merged_df, merged_data_path)

if __name__ == "__main__":
    process_data()


