Tests:

Test Visulisations (test_ visulisations)

1. test_load_merged_data
Verifies if the merged dataset loads correctly.
Checks if the dataset is non-empty and contains the required columns (gdp_per_capita and aggregated_life_expectancy).

2. test_plot_scatter_with_regression
Tests if the scatter plot with a regression line is generated and saved as plot_scatter_with_regression.pdf.

3. test_plot_life_expectancy_vs_infant_mortality
Confirms the creation of the life expectancy vs. infant mortality plot saved as life_expectancy_and_infant_mortality_female_and_male.pdf.

4. test_facet_scatter_graphs_with_regression
Verifies the generation of facet scatter graphs with regression lines, saved as facet_scatter_graphs_with_regression.pdf.

5. test_regression_line_in_scatter_plot
Checks the accuracy of the regression line in the scatter plot of infant mortality vs. life expectancy using mean squared error (MSE).
MSE must be less than 50.

6. test_regression_line_in_facet_scatter_graphs
Validates regression lines in facet scatter plots for GDP and healthcare expenditure correlations.
MSE must be less than 50.

Analysis Tests (test_analysis)

1. test_summary_statistics
Validates the summary_statistics function returns a DataFrame with correct mean and variance calculations for aggregated_infant_mortality.

2. test_correlation_analysis_with_test
Checks the correctness of the correlation and hypothesis testing between aggregated_infant_mortality and aggregated_life_expectancy.
Verifies output types (float for correlation and p-value, str for result) and matches Pearson correlation results.

Data Processing Tests (Test_data_processing)

1. test_no_missing_values_in_gdp_data
Confirms there are no missing values in critical GDP data columns after cleaning.

2. test_no_missing_values_in_healthcare_data
Ensures no missing values exist in healthcare expenditure data post-cleaning.

3. test_column_consistency_in_gdp_data
Validates that cleaned GDP data retains the expected columns (entity, year, gdp_per_capita).

4. test_column_consistency_in_healthcare_data
Confirms that cleaned healthcare data includes only the intended columns.

5. test_merge_data_with_gdp_and_healthcare
Tests the merging of infant mortality, life expectancy, GDP, and healthcare datasets.
Verifies key columns are present and the merged dataset is not empty.



