import unittest
import os
import pandas as pd
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  
from src.visulisations import (
    load_merged_data,
    plot_scatter_with_regression,
    plot_life_expectancy_vs_infant_mortality,
    facet_scatter_graphs_with_regression,
    output_dir,
    merged_data_path,
)

class TestVisualisations(unittest.TestCase):
    def test_load_merged_data(self):
        """
        Test if the merged data loads correctly.
        """
        merged_df = load_merged_data(merged_data_path)
        self.assertGreater(len(merged_df), 0, "Merged data is empty.")
        self.assertIn('gdp_per_capita', merged_df.columns, "'gdp_per_capita' column not found in merged data.")
        self.assertIn('aggregated_life_expectancy', merged_df.columns, "'aggregated_life_expectancy' column not found in merged data.")

    def test_plot_scatter_with_regression(self):
        """
        Test scatter plot with regression line generation.
        """
        merged_df = load_merged_data(merged_data_path)
        output_file = f'{output_dir}/plot_scatter_with_regression.pdf'
        plot_scatter_with_regression(merged_df)
        self.assertTrue(os.path.exists(output_file), "Scatter plot with regression PDF was not created.")

    def test_plot_life_expectancy_vs_infant_mortality(self):
        """
        Test life expectancy vs infant mortality plot generation.
        """
        merged_df = load_merged_data(merged_data_path)
        output_file = f'{output_dir}/life_expectancy_and_infant_mortality_female_and_male.pdf'
        plot_life_expectancy_vs_infant_mortality(merged_df)
        self.assertTrue(os.path.exists(output_file), "Life expectancy vs infant mortality PDF was not created.")

    def test_facet_scatter_graphs_with_regression(self):
        """
        Test facet scatter graphs with regression lines generation.
        """
        merged_df = load_merged_data(merged_data_path)
        output_file = f'{output_dir}/facet_scatter_graphs_with_regression.pdf'
        facet_scatter_graphs_with_regression(merged_df)
        self.assertTrue(os.path.exists(output_file), "Facet scatter graphs with regression PDF was not created.")

    def test_regression_line_in_scatter_plot(self):
        """
        Test the regression line in the Scatter Plot of Infant Mortality vs. Life Expectancy.
        """
        merged_df = load_merged_data(merged_data_path)
        X = merged_df[['aggregated_infant_mortality']]
        y = merged_df['aggregated_life_expectancy']

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)

        self.assertLess(mse, 50, f"Mean squared error is too high: {mse}")
        self.assertEqual(len(y_pred), len(y), "The number of predicted values does not match the actual values.")

    def test_regression_line_in_facet_scatter_graphs(self):
        """
        Test regression lines in the facet scatter graphs for GDP and Healthcare.
        """
        merged_df = load_merged_data(merged_data_path)

        X_gdp = merged_df[['gdp_per_capita']]
        y_life = merged_df['aggregated_life_expectancy']
        model_gdp = LinearRegression()
        model_gdp.fit(X_gdp, y_life)
        mse_gdp = mean_squared_error(y_life, model_gdp.predict(X_gdp))
        self.assertLess(mse_gdp, 50, f"Mean squared error for Life Expectancy vs GDP is too high: {mse_gdp}")

        X_healthcare = merged_df[['health_expenditure_per_capita_-_total']]
        y_mortality = merged_df['aggregated_infant_mortality']
        model_healthcare = LinearRegression()
        model_healthcare.fit(X_healthcare, y_mortality)
        mse_healthcare = mean_squared_error(y_mortality, model_healthcare.predict(X_healthcare))
        self.assertLess(mse_healthcare, 50, f"Mean squared error for Infant Mortality vs Healthcare is too high: {mse_healthcare}")

if __name__ == '__main__':
    unittest.main()