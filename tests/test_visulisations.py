import unittest
import os
import pandas as pd
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  
from src.visulisations import (
    load_merged_data,
    plot_scatter_with_regression,
    plot_life_expectancy_vs_infant_mortality,
    plot_residuals,
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

    def test_plot_residuals(self):
        """
        Test residual plot generation.
        """
        merged_df = load_merged_data(merged_data_path)
        output_file = f'{output_dir}/residual_plot.pdf'
        plot_residuals(merged_df)
        self.assertTrue(os.path.exists(output_file), "Residual plot PDF was not created.")

    def test_facet_scatter_graphs_with_regression(self):
        """
        Test facet scatter graphs with regression lines generation.
        """
        merged_df = load_merged_data(merged_data_path)
        output_file = f'{output_dir}/facet_scatter_graphs_with_regression.pdf'
        facet_scatter_graphs_with_regression(merged_df)
        self.assertTrue(os.path.exists(output_file), "Facet scatter graphs with regression PDF was not created.")

    def test_mean_calculation_in_plot_life_expectancy_vs_infant_mortality(self):
        """
        Test if the mean values are calculated accurately when grouping by year.
        """
        merged_df = load_merged_data(merged_data_path)
        grouped_df = merged_df.groupby('year')[
            [
                'period_life_expectancy_-_sex:_female_-_age:_0',
                'period_life_expectancy_-_sex:_male_-_age:_0',
                'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
                'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
            ]
        ].mean()

        for year in grouped_df.index:
            # Calculate expected mean manually for a sample year
            sample_year_df = merged_df[merged_df['year'] == year]
            expected_female_life_expectancy = sample_year_df['period_life_expectancy_-_sex:_female_-_age:_0'].mean()
            actual_female_life_expectancy = grouped_df.loc[year, 'period_life_expectancy_-_sex:_female_-_age:_0']
            self.assertAlmostEqual(
                expected_female_life_expectancy, 
                actual_female_life_expectancy,
                places=5,
                msg=f"Mean calculation for female life expectancy in year {year} is incorrect."
            )

    def test_plot_life_expectancy_vs_infant_mortality(self):
        """
        Test life expectancy vs infant mortality plot generation and ensure critical data is plotted.
        """
        merged_df = load_merged_data(merged_data_path)
        output_file = f'{output_dir}/life_expectancy_and_infant_mortality_female_and_male.pdf'

        # Generate the plot
        plot_life_expectancy_vs_infant_mortality(merged_df)

        # Check if the file is created
        self.assertTrue(os.path.exists(output_file), "Life expectancy vs infant mortality PDF was not created.")

        # Check critical columns are present in the grouped data used for plotting
        grouped_df = merged_df.groupby('year')[
            [
                'period_life_expectancy_-_sex:_female_-_age:_0',
                'period_life_expectancy_-_sex:_male_-_age:_0',
                'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
                'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
            ]
        ].mean()

        # Ensure the grouped dataframe is not empty and contains correct data
        self.assertGreater(len(grouped_df), 0, "Grouped data used for plotting is empty.")
        self.assertTrue(
            all(col in grouped_df.columns for col in [
                'period_life_expectancy_-_sex:_female_-_age:_0',
                'period_life_expectancy_-_sex:_male_-_age:_0',
                'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
                'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
            ]),
            "Critical columns are missing in the grouped data used for plotting."
        )

    def test_regression_line_in_scatter_plot(self):
        """
        Test the regression line in the Scatter Plot of Infant Mortality vs. Life Expectancy.
        """
        merged_df = load_merged_data(merged_data_path)
        X = merged_df[['aggregated_infant_mortality']]
        y = merged_df['aggregated_life_expectancy']

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict values
        y_pred = model.predict(X)

        # Calculate mean squared error
        mse = mean_squared_error(y, y_pred)

        # Check MSE is reasonable (ensuring the model fits the data to some extent)
        self.assertLess(mse, 50, f"Mean squared error is too high: {mse}")

        # Ensure the regression line has been generated correctly
        self.assertTrue(
            len(y_pred) == len(y),
            "The number of predicted values does not match the actual values."
        )

    def test_regression_line_in_facet_scatter_graphs(self):
        """
        Test regression lines in the facet scatter graphs for GDP and Healthcare.
        """
        merged_df = load_merged_data(merged_data_path)

        # Test Life Expectancy vs GDP
        X_gdp = merged_df[['gdp_per_capita']]
        y_life = merged_df['aggregated_life_expectancy']
        model_gdp = LinearRegression()
        model_gdp.fit(X_gdp, y_life)
        y_pred_gdp = model_gdp.predict(X_gdp)
        mse_gdp = mean_squared_error(y_life, y_pred_gdp)
        self.assertLess(mse_gdp, 50, f"Mean squared error for Life Expectancy vs GDP is too high: {mse_gdp}")

        # Test Infant Mortality vs Healthcare
        X_healthcare = merged_df[['health_expenditure_per_capita_-_total']]
        y_mortality = merged_df['aggregated_infant_mortality']
        model_healthcare = LinearRegression()
        model_healthcare.fit(X_healthcare, y_mortality)
        y_pred_healthcare = model_healthcare.predict(X_healthcare)
        mse_healthcare = mean_squared_error(y_mortality, y_pred_healthcare)
        self.assertLess(mse_healthcare, 50, f"Mean squared error for Infant Mortality vs Healthcare is too high: {mse_healthcare}")

    def test_facet_scatter_graphs_with_regression_output(self):
        """
        Test facet scatter graphs with regression lines are generated and saved.
        """
        merged_df = load_merged_data(merged_data_path)
        output_file = f'{output_dir}/facet_scatter_graphs_with_regression.pdf'

        # Generate the plot
        facet_scatter_graphs_with_regression(merged_df)

        # Check if the file is created
        self.assertTrue(os.path.exists(output_file), "Facet scatter graphs with regression PDF was not created.")

if __name__ == '__main__':
    unittest.main()