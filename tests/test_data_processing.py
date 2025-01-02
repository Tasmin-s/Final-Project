import pytest
import pandas as pd
from src.data_processing import load_data, clean_data, merge_data

# Test for missing values in critical columns
def test_no_missing_values_in_infant_mortality_data():
    infant_mortality_df = load_data('/Users/Tasmin/Final-Project/data/raw/infant-mortality-rate-wdi.csv')  
    # Use the normalized column names
    columns_to_check = [
        'entity', 'year',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
    ]
    cleaned_data = clean_data(infant_mortality_df, columns_to_check)
    
    # Ensure there are no missing values in critical columns
    assert cleaned_data[columns_to_check].isnull().sum().sum() == 0, "Missing values found in critical columns."

# Test for column consistency
def test_column_consistency_in_infant_mortality_data():
    infant_mortality_df = load_data('/Users/Tasmin/Final-Project/data/raw/infant-mortality-rate-wdi.csv')
    
    # Use the normalized column names
    columns_to_check = [
        'entity', 'year',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
    ]
    
    cleaned_data = clean_data(infant_mortality_df, columns_to_check)
    
    # Drop the 'code' column from cleaned data if it exists
    cleaned_data = cleaned_data.drop(columns=['code'], errors='ignore')
    
    # Define the expected columns after cleaning
    expected_columns = [
        'entity', 'year',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
    ]
    
    # Ensure that the cleaned data contains the expected columns
    assert list(cleaned_data.columns) == expected_columns, f"Columns do not match. Found: {cleaned_data.columns}, Expected: {expected_columns}"

# Test for valid data ranges (e.g., infant mortality rate should be positive)
def test_valid_data_ranges_in_infant_mortality_data():
    infant_mortality_df = load_data('/Users/Tasmin/Final-Project/data/raw/infant-mortality-rate-wdi.csv')  
    # Use the normalized column names
    columns_to_check = [
        'entity', 'year',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
    ]
    cleaned_data = clean_data(infant_mortality_df, columns_to_check)
    
    # Ensure that the values in the relevant columns are positive
    mortality_female = cleaned_data[
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
    ]
    mortality_male = cleaned_data[
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
    ]
    
    assert (mortality_female > 0).all(), "Some female mortality rates are non-positive."
    assert (mortality_male > 0).all(), "Some male mortality rates are non-positive."

# Test for merging data
def test_merge_data():
    infant_mortality_df = load_data('/Users/Tasmin/Final-Project/data/raw/infant-mortality-rate-wdi.csv')
    life_expectancy_df = load_data('/Users/Tasmin/Final-Project/data/raw/life-expectation-at-birth-by-sex.csv')
    
    columns_to_check_infant_mortality = [
        'entity', 'year',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
        'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
    ]
    columns_to_check_life_expectancy = [
        'entity', 'year',
        'period_life_expectancy_-_sex:_female_-_age:_0',
        'period_life_expectancy_-_sex:_male_-_age:_0'
    ]
    
    cleaned_infant_mortality_data = clean_data(infant_mortality_df, columns_to_check_infant_mortality)
    cleaned_life_expectancy_data = clean_data(life_expectancy_df, columns_to_check_life_expectancy)
    
    # Merge the datasets
    merged_data = merge_data(cleaned_infant_mortality_data, cleaned_life_expectancy_data)
    
    # Check if the 'entity' and 'year' columns are common in the merged data
    assert 'entity' in merged_data.columns, "'entity' column not found in merged data."
    assert 'year' in merged_data.columns, "'year' column not found in merged data."
    assert len(merged_data) > 0, "Merged data is empty."
    
    # Print the columns of merged_data, not cleaned_data
    print(merged_data.columns)
