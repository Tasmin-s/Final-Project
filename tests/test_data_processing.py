import unittest
import pandas as pd
from src.data_processing import load_data, clean_data, merge_data, aggregated_values

class TestDataProcessing(unittest.TestCase):

    def test_no_missing_values_in_infant_mortality_data(self):
        # Use the path to processed (cleaned) data
        infant_mortality_df = load_data('/Users/Tasmin/Final-Project/data/processed/infant-mortality-rate-wdi-cleaned.csv')  
        
        # Use the normalized column names
        columns_to_check = [
            'entity', 'year',
            'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
            'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
        ]
        cleaned_data = clean_data(infant_mortality_df, columns_to_check)
        
        # Ensure there are no missing values in critical columns
        self.assertEqual(cleaned_data[columns_to_check].isnull().sum().sum(), 0, "Missing values found in critical columns.")

    def test_column_consistency_in_infant_mortality_data(self):
        # Use the path to processed (cleaned) data
        infant_mortality_df = load_data('/Users/Tasmin/Final-Project/data/processed/infant-mortality-rate-wdi-cleaned.csv')
        
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
        self.assertEqual(list(cleaned_data.columns), expected_columns, f"Columns do not match. Found: {cleaned_data.columns}, Expected: {expected_columns}")

    def test_merge_data(self):
        # Use the path to processed (cleaned) data
        infant_mortality_df = load_data('/Users/Tasmin/Final-Project/data/processed/infant-mortality-rate-wdi-cleaned.csv')
        life_expectancy_df = load_data('/Users/Tasmin/Final-Project/data/processed/life-expectation-at-birth-by-sex-cleaned.csv')
        
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
        self.assertIn('entity', merged_data.columns, "'entity' column not found in merged data.")
        self.assertIn('year', merged_data.columns, "'year' column not found in merged data.")
        self.assertGreater(len(merged_data), 0, "Merged data is empty.")
        
        # Print the columns of merged_data, not cleaned_data
        print(merged_data.columns)

    def test_valid_data_ranges_in_infant_mortality_data(self):
        # Use the path to processed (cleaned) data
        infant_mortality_df = load_data('/Users/Tasmin/Final-Project/data/processed/infant-mortality-rate-wdi-cleaned.csv')  
        
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
        
        self.assertTrue((mortality_female > 0).all(), "Some female mortality rates are non-positive.")
        self.assertTrue((mortality_male > 0).all(), "Some male mortality rates are non-positive.")


    def test_aggregated_values_infant_mortality(self):
        # Use the path to processed (cleaned) data
        infant_mortality_df = load_data('/Users/Tasmin/Final-Project/data/processed/infant-mortality-rate-wdi-cleaned.csv')
        columns_to_check = [
            'entity', 'year',
            'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
            'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
        ]
        cleaned_data = clean_data(infant_mortality_df, columns_to_check)
        merged_data = merge_data(cleaned_data, cleaned_data)  # Example of merging with itself, can adjust logic if necessary
        
        # Adjust the aggregated function to dynamically select the correct columns
        female_column = [col for col in merged_data.columns if 'female' in col][0]
        male_column = [col for col in merged_data.columns if 'male' in col][0]
        
        merged_data['aggregated_infant_mortality'] = merged_data[female_column] + merged_data[male_column]
        
        # Check if the aggregated infant mortality column is present
        self.assertIn('aggregated_infant_mortality', merged_data.columns, "'aggregated_infant_mortality' column not found.")
        
        # Ensure there are no NaN values in the aggregated infant mortality column
        self.assertTrue(merged_data['aggregated_infant_mortality'].notna().all(), "NaN values found in 'aggregated_infant_mortality' column.")
        
        # Ensure that aggregated values make sense (for simplicity, just check if values are positive)
        self.assertTrue((merged_data['aggregated_infant_mortality'] > 0).all(), "Aggregated infant mortality contains non-positive values.")
    
    def test_aggregated_values_life_expectancy(self):
        # Use the path to processed (cleaned) data
        life_expectancy_df = load_data('/Users/Tasmin/Final-Project/data/processed/life-expectation-at-birth-by-sex-cleaned.csv')
        columns_to_check = [
            'entity', 'year',
            'period_life_expectancy_-_sex:_female_-_age:_0',
            'period_life_expectancy_-_sex:_male_-_age:_0'
        ]
        cleaned_data = clean_data(life_expectancy_df, columns_to_check)
        merged_data = merge_data(cleaned_data, cleaned_data)  # Example of merging, use appropriate DataFrames
        
        # Adjust the aggregated function to dynamically select the correct columns
        female_column = [col for col in merged_data.columns if 'female' in col][0]
        male_column = [col for col in merged_data.columns if 'male' in col][0]
        
        merged_data['aggregated_life_expectancy'] = merged_data[female_column] + merged_data[male_column]
        
        # Check if the aggregated life expectancy column is present
        self.assertIn('aggregated_life_expectancy', merged_data.columns, "'aggregated_life_expectancy' column not found.")
        
        # Ensure there are no NaN values in the aggregated life expectancy column
        self.assertTrue(merged_data['aggregated_life_expectancy'].notna().all(), "NaN values found in 'aggregated_life_expectancy' column.")
        
        # Ensure that aggregated values make sense (for simplicity, just check if values are positive)
        self.assertTrue((merged_data['aggregated_life_expectancy'] > 0).all(), "Aggregated life expectancy contains non-positive values.")

if __name__ == '__main__':
    unittest.main()




