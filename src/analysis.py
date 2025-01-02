import pandas as pd
import statsmodels.api as sm

# 1. Compute Summary Statistics for Infant Mortality and Life Expectancy
def summary_statistics(df, columns):
    """
    Function to calculate summary statistics for specified columns.
    """
    summary = df[columns].describe().transpose()
    summary['variance'] = df[columns].var()
    return summary

# 2. Calculate Correlation Between Infant Mortality and Life Expectancy
def correlation_analysis(df, mortality_column, life_expectancy_column):
    """
    Function to compute the correlation between infant mortality rate and life expectancy.
    """
    correlation = df[[mortality_column, life_expectancy_column]].corr().iloc[0, 1]
    return correlation

# 3. Analyze Trends Over Time
def trends_over_time(df, mortality_column, life_expectancy_column, group_by_column='year'):
    """
    Function to analyze trends in infant mortality and life expectancy over time.
    """
    trends = df.groupby(group_by_column)[[mortality_column, life_expectancy_column]].mean()
    trends['mortality_rate_change'] = trends[mortality_column].pct_change()
    trends['life_expectancy_change'] = trends[life_expectancy_column].pct_change()
    return trends

# 4. Perform Regression Analysis to Understand Relationships Between Variables
def regression_analysis(df, y_column, X_columns):
    """
    Function to perform linear regression analysis to assess the impact of various factors.
    """
    X = df[X_columns]
    X = sm.add_constant(X)  # Adds an intercept to the model
    y = df[y_column]

    model = sm.OLS(y, X).fit()  # Fit the model
    return model.summary()

# 5. Save Analysis Results to CSV
def save_results_to_csv(df, file_path):
    """
    Save the results of the analysis to a CSV file.
    """
    df.to_csv(file_path, index=False)

if __name__ == '__main__':
    # File paths
    merged_data_path = '/Users/Tasmin/Final-Project/data/processed/merged_data.csv'
    trends_output_path = '/Users/Tasmin/Final-Project/data/processed/trends.csv'
    stats_output_path = '/Users/Tasmin/Final-Project/data/processed/summary_statistics.csv'

    # Load the preprocessed and merged data
    print("Loading Processed Data...")
    merged_data = pd.read_csv(merged_data_path)

    # Debug merged DataFrame columns
    print("Merged DataFrame Columns:")
    print(merged_data.columns.tolist())

    # Define aggregated columns for analysis
    aggregated_infant_mortality_column = 'aggregated_infant_mortality'
    aggregated_life_expectancy_column = 'aggregated_life_expectancy'

    # Summary statistics
    print("Calculating Summary Statistics...")
    infant_mortality_stats = summary_statistics(merged_data, [aggregated_infant_mortality_column])
    life_expectancy_stats = summary_statistics(merged_data, [aggregated_life_expectancy_column])
    print(infant_mortality_stats)
    print(life_expectancy_stats)

    # Save summary statistics
    summary_stats_df = pd.concat([infant_mortality_stats, life_expectancy_stats])
    save_results_to_csv(summary_stats_df, stats_output_path)

    # Correlation analysis
    print("Performing Correlation Analysis...")
    correlation = correlation_analysis(merged_data, aggregated_infant_mortality_column, aggregated_life_expectancy_column)
    print(f"Correlation between Infant Mortality and Life Expectancy: {correlation}")

    # Trends over time
    print("Analyzing Trends Over Time...")
    trends = trends_over_time(merged_data, aggregated_infant_mortality_column, aggregated_life_expectancy_column)
    print(trends)

    # Save trends
    save_results_to_csv(trends, trends_output_path)

    # Regression analysis
    print("Performing Regression Analysis...")
    regression_result = regression_analysis(
        merged_data,
        aggregated_infant_mortality_column,
        ['year', aggregated_life_expectancy_column]
    )
    print(regression_result)

    
