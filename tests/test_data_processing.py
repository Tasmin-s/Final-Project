import unittest
import pandas as pd
from src.data_processing import load_data, clean_data, merge_data, aggregated_values

class TestDataProcessing(unittest.TestCase):

    def test_no_missing_values_in_gdp_data(self):
        gdp_life_df = load_data('./data/processed/life-expectancy-vs-gdp-per-capita-cleaned.csv')
        columns_to_check = [
            'entity', 'year', 'gdp_per_capita'
        ]
        cleaned_data = clean_data(gdp_life_df, columns_to_check)
        self.assertEqual(cleaned_data[columns_to_check].isnull().sum().sum(), 0, "Missing values found in critical columns for GDP data.")

    def test_no_missing_values_in_healthcare_data(self):
        healthcare_df = load_data('./data/processed/life-expectancy-vs-health-expenditure-cleaned.csv')
        columns_to_check = [
            'entity', 'year', 'health_expenditure_per_capita_-_total'
        ]
        cleaned_data = clean_data(healthcare_df, columns_to_check)
        self.assertEqual(cleaned_data[columns_to_check].isnull().sum().sum(), 0, "Missing values found in critical columns for healthcare data.")

    def test_column_consistency_in_gdp_data(self):
        gdp_life_df = load_data('./data/processed/life-expectancy-vs-gdp-per-capita-cleaned.csv')
        columns_to_check = [
            'entity', 'year', 'gdp_per_capita'
        ]
        cleaned_data = clean_data(gdp_life_df, columns_to_check)
        expected_columns = ['entity', 'year', 'gdp_per_capita']
        self.assertEqual(list(cleaned_data.columns), expected_columns, f"Columns do not match for GDP data. Found: {cleaned_data.columns}, Expected: {expected_columns}")

    def test_column_consistency_in_healthcare_data(self):
        healthcare_df = load_data('./data/processed/life-expectancy-vs-health-expenditure-cleaned.csv')
        columns_to_check = [
            'entity', 'year', 'health_expenditure_per_capita_-_total'
        ]
        cleaned_data = clean_data(healthcare_df, columns_to_check)
        expected_columns = ['entity', 'year', 'health_expenditure_per_capita_-_total']
        self.assertEqual(list(cleaned_data.columns), expected_columns, f"Columns do not match for healthcare data. Found: {cleaned_data.columns}, Expected: {expected_columns}")

    def test_merge_data_with_gdp_and_healthcare(self):
        infant_mortality_df = load_data('./data/processed/infant-mortality-rate-wdi-cleaned.csv')
        life_expectancy_df = load_data('./data/processed/life-expectation-at-birth-by-sex-cleaned.csv')
        gdp_life_df = load_data('./data/processed/life-expectancy-vs-gdp-per-capita-cleaned.csv')
        healthcare_df = load_data('./data/processed/life-expectancy-vs-health-expenditure-cleaned.csv')

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
        columns_to_check_gdp = ['entity', 'year', 'gdp_per_capita']
        columns_to_check_healthcare = ['entity', 'year', 'health_expenditure_per_capita_-_total']

        cleaned_infant_mortality_data = clean_data(infant_mortality_df, columns_to_check_infant_mortality)
        cleaned_life_expectancy_data = clean_data(life_expectancy_df, columns_to_check_life_expectancy)
        cleaned_gdp_data = clean_data(gdp_life_df, columns_to_check_gdp)
        cleaned_healthcare_data = clean_data(healthcare_df, columns_to_check_healthcare)

        merged_data = merge_data(cleaned_infant_mortality_data, cleaned_life_expectancy_data, cleaned_gdp_data, cleaned_healthcare_data)

        self.assertIn('entity', merged_data.columns, "'entity' column not found in merged data.")
        self.assertIn('year', merged_data.columns, "'year' column not found in merged data.")
        self.assertIn('gdp_per_capita', merged_data.columns, "'gdp_per_capita' column not found in merged data.")
        self.assertIn('health_expenditure_per_capita_-_total', merged_data.columns, "'health_expenditure_per_capita_-_total' column not found in merged data.")
        self.assertGreater(len(merged_data), 0, "Merged data is empty.")

if __name__ == '__main__':
    unittest.main()






