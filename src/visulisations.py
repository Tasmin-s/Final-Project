import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the path to the merged data
merged_data_path = '/Users/Tasmin/Final-Project/data/processed/merged_data.csv'

# Define the output directory
output_dir = '/Users/Tasmin/Final-Project/figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_merged_data(file_path):
    """
    Load the merged dataset from the specified path.
    """
    return pd.read_csv(file_path)

def plot_scatter_with_regression(merged_df):
    """
    Scatter Plot with Regression Line: Infant mortality vs. life expectancy.
    """
    plt.figure(figsize=(12, 8))

    # Scatter plot
    sns.scatterplot(
        x='aggregated_infant_mortality', 
        y='aggregated_life_expectancy', 
        data=merged_df, 
        color='blue', alpha=0.7, label='Data Points'
    )

    # Fit and plot regression line
    sns.regplot(
        x='aggregated_infant_mortality', 
        y='aggregated_life_expectancy', 
        data=merged_df, 
        scatter=False, color='red', label='Regression Line'
    )

    # Add labels, title, and legend
    plt.title('Scatter Plot of Infant Mortality vs. Life Expectancy', fontsize=16)
    plt.xlabel('Aggregated Infant Mortality (deaths per 100 live births)', fontsize=14)
    plt.ylabel('Aggregated Life Expectancy (years)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Add caption
    plt.figtext(0.5, -0.05, 'This scatter plot shows the relationship between infant mortality rates and life expectancy over time, with a regression line to illustrate the trend.', wrap=True, horizontalalignment='center', fontsize=12)

    # Save the plot as a PDF in the figures directory
    plt.savefig(f'{output_dir}/plot_scatter_with_regression.pdf')
    plt.close()

def plot_life_expectancy_vs_infant_mortality(merged_df):
    """
    Plot Life Expectancy vs Infant Mortality for Female and Male, and save it as a PDF.
    """
    df_grouped = merged_df.groupby('year')[
        [
            'period_life_expectancy_-_sex:_female_-_age:_0', 
            'period_life_expectancy_-_sex:_male_-_age:_0',
            'observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births',
            'observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'
        ]
    ].mean().reset_index()

    # Create the plot
    plt.figure(figsize=(14, 8))
    ax1 = plt.gca()

    # Plot life expectancy for females and males
    ax1.plot(
        df_grouped['year'], 
        df_grouped['period_life_expectancy_-_sex:_female_-_age:_0'], 
        color='blue', label='Female Life Expectancy'
    )
    ax1.plot(
        df_grouped['year'], 
        df_grouped['period_life_expectancy_-_sex:_male_-_age:_0'], 
        color='green', label='Male Life Expectancy'
    )
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Life Expectancy', fontsize=14, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for infant mortality
    ax2 = ax1.twinx()
    ax2.plot(
        df_grouped['year'], 
        df_grouped['observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'], 
        color='orange', linestyle='--', label='Female Infant Mortality'
    )
    ax2.plot(
        df_grouped['year'], 
        df_grouped['observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'], 
        color='red', linestyle='--', label='Male Infant Mortality'
    )
    ax2.set_ylabel('Infant Mortality Rate (per 100 live births)', fontsize=14, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add a title and legend
    plt.title('Life Expectancy vs Infant Mortality Over Time (By Sex)', fontsize=16)
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    # Add a caption
    caption = (
        "This plot compares the trends in life expectancy and infant mortality rates for both "
        "females and males over time. The life expectancy is plotted as solid lines, while the "
        "infant mortality rates are represented as dashed lines. The comparison reveals how these "
        "two factors have evolved in relation to each other, potentially shedding light on public "
        "health trends and their impact on societal well-being."
    )
    plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=12)

    plt.tight_layout()

    # Define the output directory for saving
    output_dir = '/Users/Tasmin/Final-Project/figures'
    
    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot as a PDF in the figures directory
    plt.savefig(f'{output_dir}/life_expectancy_and_infant_mortality_female_and_male.pdf')
    
    # Close the plot 
    plt.close()

def plot_residuals(merged_df):
    """
    Plot residuals of the linear regression model: Predicted vs Actual values.
    """
    # Prepare data for regression model
    X = merged_df[['aggregated_infant_mortality']]  # Independent variable
    y = merged_df['aggregated_life_expectancy']    # Dependent variable
    
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict values using the regression model
    y_pred = model.predict(X)
    
    # Calculate residuals (difference between actual and predicted values)
    residuals = y - y_pred

    # Create a residual plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=y_pred, y=residuals, color='blue', alpha=0.7)
    
    # Plot the horizontal line for zero residuals
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Add labels and title
    plt.title('Residual Plot: Predicted vs Actual Values', fontsize=16)
    plt.xlabel('Predicted Life Expectancy', fontsize=14)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=14)
    
    plt.tight_layout()

    # Save the plot as a PDF in the figures directory
    plt.savefig(f'{output_dir}/residual_plot.pdf')
    plt.close()

def facet_scatter_graphs_with_regression(df):
    """
    Create four scatter plots with regression lines:
    - Life Expectancy vs GDP
    - Life Expectancy vs Healthcare Investment
    - Infant Mortality vs GDP
    - Infant Mortality vs Healthcare Investment
    Each plot includes a caption explaining the trends.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Life Expectancy vs GDP
    sns.regplot(
        x='gdp_per_capita',
        y='aggregated_life_expectancy',
        data=df,
        ax=axes[0, 0],
        scatter_kws={'alpha': 0.7},
        line_kws={'color': 'blue'}
    )
    axes[0, 0].set_title('Life Expectancy vs GDP', fontsize=16)
    axes[0, 0].set_xlabel('GDP per Capita (USD)', fontsize=14)
    axes[0, 0].set_ylabel('Life Expectancy (Years)', fontsize=14)

    # Life Expectancy vs Healthcare Investment
    sns.regplot(
        x='health_expenditure_per_capita_-_total',
        y='aggregated_life_expectancy',
        data=df,
        ax=axes[0, 1],
        scatter_kws={'alpha': 0.7},
        line_kws={'color': 'blue'}
    )
    axes[0, 1].set_title('Life Expectancy vs Healthcare Investment', fontsize=16)
    axes[0, 1].set_xlabel('Healthcare Investment (USD)', fontsize=14)
    axes[0, 1].set_ylabel('Life Expectancy (Years)', fontsize=14)

    # Infant Mortality vs GDP
    sns.regplot(
        x='gdp_per_capita',
        y='aggregated_infant_mortality',
        data=df,
        ax=axes[1, 0],
        scatter_kws={'alpha': 0.7, 'color': 'red'},
        line_kws={'color': 'red'}
    )
    axes[1, 0].set_title('Infant Mortality vs GDP', fontsize=16)
    axes[1, 0].set_xlabel('GDP per Capita (USD)', fontsize=14)
    axes[1, 0].set_ylabel('Infant Mortality (per 100 live births)', fontsize=14)

    # Infant Mortality vs Healthcare Investment
    sns.regplot(
        x='health_expenditure_per_capita_-_total',
        y='aggregated_infant_mortality',
        data=df,
        ax=axes[1, 1],
        scatter_kws={'alpha': 0.7, 'color': 'red'},
        line_kws={'color': 'red'}
    )
    axes[1, 1].set_title('Infant Mortality vs Healthcare Investment', fontsize=16)
    axes[1, 1].set_xlabel('Healthcare Investment (USD)', fontsize=14)
    axes[1, 1].set_ylabel('Infant Mortality (per 100 live births)', fontsize=14)

    # Add captions
    caption = (
        "These scatter plots examine the relationships between GDP, healthcare investment, "
        "life expectancy, and infant mortality. Regression lines indicate trends, showing "
        "how economic and healthcare factors impact public health outcomes."
    )
    fig.text(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/facet_scatter_graphs_with_regression.pdf')
    plt.close()


def main():
    """
    Main function to load data and generate visualizations.
    """
    # Load the merged data
    merged_df = load_merged_data(merged_data_path)

    # Generate visualizations
    print("Generating Scatter Plot with Regression Line...")
    plot_scatter_with_regression(merged_df)

    print("Generating Life Expectancy and Infant Mortality Plot for Female and Male...")
    plot_life_expectancy_vs_infant_mortality(merged_df)

    print("Generating Residual Plot...")
    plot_residuals(merged_df)

    # Generate scatter graphs with regression lines
    print("Generating Scatter Graphs with Regression Lines...")
    facet_scatter_graphs_with_regression(merged_df)



if __name__ == '__main__':
    main()
