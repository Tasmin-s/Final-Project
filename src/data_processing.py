import pandas as pd
import os

def load_data(file_path):
    """
    Load raw data from a specified file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)

def clean_data(df, columns_to_check, essential_columns=None):
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

    # Retain only essential columns if specified
    if essential_columns is not None:
        df = df[essential_columns]
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned data to the specified output path.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

def merge_data(infant_mortality_df, life_expectancy_df, gdp_life_df, healthcare_df):
    """
    Merge the infant mortality, life expectancy, GDP, and healthcare datasets on 'year' and 'entity'.
    """
    # Merge infant mortality and life expectancy datasets
    merged_df = pd.merge(infant_mortality_df, life_expectancy_df, on=['year', 'entity'], how='inner')

    # Merge the result with the GDP dataset
    merged_df = pd.merge(merged_df, gdp_life_df, on=['year', 'entity'], how='inner')

    # Merge the result with the healthcare dataset
    merged_df = pd.merge(merged_df, healthcare_df, on=['year', 'entity'], how='inner')
    
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
    Main function to load, clean, merge, and save data for all datasets.
    """
    """
    Main function to load, clean, merge, and save data for all datasets.
    """
    # Define paths for the raw data files
    infant_mortality_raw_path = './data/raw/infant-mortality-rate-wdi.csv'  
    life_expectancy_raw_path = './data/raw/life-expectation-at-birth-by-sex.csv'
    gdp_life_raw_path = './data/raw/life-expectancy-vs-gdp-per-capita.csv'
    healthcare_raw_path = './data/raw/life-expectancy-vs-health-expenditure.csv'  
    
    # Define paths for the output (cleaned) data files
    infant_mortality_cleaned_path = './data/processed/infant-mortality-rate-wdi-cleaned.csv'
    life_expectancy_cleaned_path = './data/processed/life-expectation-at-birth-by-sex-cleaned.csv'  
    gdp_life_cleaned_path = './data/processed/life-expectancy-vs-gdp-per-capita-cleaned.csv'
    healthcare_cleaned_path = './data/processed/life-expectancy-vs-health-expenditure-cleaned.csv'
    merged_data_path = './data/processed/merged_data.csv'
    
    # Columns to check for missing values in infant mortality data
    infant_mortality_columns = [
        'year',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
    ]
    
    # Columns to check for missing values in life expectancy data
    life_expectancy_columns = [
        'year',
        'period_life_expectancy_-_sex:_female_-_age:_0',
        'period_life_expectancy_-_sex:_male_-_age:_0'
    ]

    # Columns to check for missing values in GDP data
    gdp_life_columns = [
        'year',
        'gdp_per_capita'
    ]

    # Columns to check for missing values in healthcare expenditure data
    healthcare_columns = [
        'year',
        'health_expenditure_per_capita_-_total'
    ]
    
    # Load, clean, and save data for infant mortality
    print("Processing Infant Mortality Data...")
    infant_mortality_df = load_data(infant_mortality_raw_path)
    cleaned_infant_mortality_df = clean_data(infant_mortality_df, infant_mortality_columns, essential_columns=['entity', 'year', 'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births', 'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'])
    save_cleaned_data(cleaned_infant_mortality_df, infant_mortality_cleaned_path)

    # Load, clean, and save data for GDP per capita
    print("Processing GDP Data...")
    gdp_life_df = load_data(gdp_life_raw_path)
    cleaned_gdp_life_df = clean_data(gdp_life_df, gdp_life_columns, essential_columns=['entity', 'year', 'gdp_per_capita'])
    save_cleaned_data(cleaned_gdp_life_df, gdp_life_cleaned_path)

    # Load, clean, and save data for healthcare expenditure
    print("Processing Healthcare Data...")
    healthcare_df = load_data(healthcare_raw_path)
    cleaned_healthcare_df = clean_data(healthcare_df, healthcare_columns, essential_columns=['entity', 'year', 'health_expenditure_per_capita_-_total'])
    save_cleaned_data(cleaned_healthcare_df, healthcare_cleaned_path)
    
    # Load, clean, and save data for life expectancy
    print("Processing Life Expectancy Data...")
    life_expectancy_df = load_data(life_expectancy_raw_path)
    cleaned_life_expectancy_df = clean_data(life_expectancy_df, life_expectancy_columns, essential_columns=['entity', 'year', 'period_life_expectancy_-_sex:_female_-_age:_0', 'period_life_expectancy_-_sex:_male_-_age:_0'])
    save_cleaned_data(cleaned_life_expectancy_df, life_expectancy_cleaned_path)
    
    # Merge the cleaned data on 'year' and 'entity'
    print("Merging Data...")
    merged_df = merge_data(cleaned_infant_mortality_df, cleaned_life_expectancy_df, cleaned_gdp_life_df, cleaned_healthcare_df)
    
    # Apply aggregation
    print("Applying Aggregation...")
    merged_df = aggregated_values(merged_df)
    
    # Save merged data
    save_cleaned_data(merged_df, merged_data_path)

if __name__ == "__main__":
    process_data()


