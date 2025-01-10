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
merged_data_path = './data/processed/merged_data.csv'

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
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Add detailed caption
    plt.figtext(0.5, 0.01, 'Figure 1: This scatter plot illustrates the relationship between aggregated infant mortality rates and life expectancy. The red regression line highlights the overall negative trend, indicating that higher infant mortality is associated with lower life expectancy.', wrap=True, horizontalalignment='center', fontsize=12)

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
    ax1.plot(df_grouped['year'], df_grouped['period_life_expectancy_-_sex:_female_-_age:_0'], color='blue', label='Female Life Expectancy')
    ax1.plot(df_grouped['year'], df_grouped['period_life_expectancy_-_sex:_male_-_age:_0'], color='green', label='Male Life Expectancy')
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Life Expectancy', fontsize=14, color='blue')

    # Create a secondary y-axis for infant mortality
    ax2 = ax1.twinx()
    ax2.plot(df_grouped['year'], df_grouped['observation_value_-_indicator:_infant_mortality_rate_-_sex:_female_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'], color='orange', linestyle='--', label='Female Infant Mortality')
    ax2.plot(df_grouped['year'], df_grouped['observation_value_-_indicator:_infant_mortality_rate_-_sex:_male_-_wealth_quintile:_total_-_unit_of_measure:_deaths_per_100_live_births'], color='red', linestyle='--', label='Male Infant Mortality')
    ax2.set_ylabel('Infant Mortality Rate (per 100 live births)', fontsize=14, color='red')

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Add detailed caption
    plt.figtext(0.5, 0.01, 'Figure 2: This plot compares life expectancy and infant mortality rates for males and females over the years. Solid lines indicate life expectancy, while dashed lines represent infant mortality rates, highlighting gender-based health disparities over time.', wrap=True, horizontalalignment='center', fontsize=12)

    plt.savefig(f'{output_dir}/life_expectancy_and_infant_mortality_female_and_male.pdf')
    plt.close()

def facet_scatter_graphs_with_regression(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.regplot(x='gdp_per_capita', y='aggregated_life_expectancy', data=df, ax=axes[0, 0], scatter_kws={'alpha': 0.7}, line_kws={'color': 'blue'})
    axes[0, 0].set_title('Life Expectancy vs GDP')
    axes[0, 0].set_xlabel('GDP per Capita (USD)')
    axes[0, 0].set_ylabel('Life Expectancy (Years)')
    axes[0, 0].text(0.5, -0.25, 'Figure 3: Higher GDP per capita correlates with longer life expectancy.', transform=axes[0, 0].transAxes, ha='center', fontsize=10, wrap=True)

    sns.regplot(x='health_expenditure_per_capita_-_total', y='aggregated_life_expectancy', data=df, ax=axes[0, 1], scatter_kws={'alpha': 0.7}, line_kws={'color': 'blue'})
    axes[0, 1].set_title('Life Expectancy vs Healthcare Investment')
    axes[0, 1].set_xlabel('Healthcare Investment (USD)')
    axes[0, 1].set_ylabel('Life Expectancy (Years)')
    axes[0, 1].text(0.5, -0.25, 'Figure 4: More healthcare spending leads to longer life expectancy.', transform=axes[0, 1].transAxes, ha='center', fontsize=10, wrap=True)

    sns.regplot(x='gdp_per_capita', y='aggregated_infant_mortality', data=df, ax=axes[1, 0], scatter_kws={'alpha': 0.7, 'color': 'red'}, line_kws={'color': 'red'})
    axes[1, 0].set_title('Infant Mortality vs GDP')
    axes[1, 0].set_xlabel('GDP per Capita (USD)')
    axes[1, 0].set_ylabel('Infant Mortality (per 100 live births)')
    axes[1, 0].text(0.5, -0.25, 'Figure 5: Higher GDP per capita lowers infant mortality.', transform=axes[1, 0].transAxes, ha='center', fontsize=10, wrap=True)

    sns.regplot(x='health_expenditure_per_capita_-_total', y='aggregated_infant_mortality', data=df, ax=axes[1, 1], scatter_kws={'alpha': 0.7, 'color': 'red'}, line_kws={'color': 'red'})
    axes[1, 1].set_title('Infant Mortality vs Healthcare Investment')
    axes[1, 1].set_xlabel('Healthcare Investment (USD)')
    axes[1, 1].set_ylabel('Infant Mortality (per 100 live births)')
    axes[1, 1].text(0.5, -0.25, 'Figure 6: More healthcare spending reduces infant mortality.', transform=axes[1, 1].transAxes, ha='center', fontsize=10, wrap=True)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(f'{output_dir}/facet_scatter_graphs_with_regression.pdf')
    plt.close()

def main():
    """
    Main function to load data and generate visualizations.
    """
    # Load the merged data
    merged_df = load_merged_data(merged_data_path)

    # Generate visualisations
    print("Generating Scatter Plot with Regression Line...")
    plot_scatter_with_regression(merged_df)

    print("Generating Life Expectancy and Infant Mortality Plot for Female and Male...")
    plot_life_expectancy_vs_infant_mortality(merged_df)

    print("Generating Scatter Graphs with Regression Lines...")
    facet_scatter_graphs_with_regression(merged_df)

if __name__ == '__main__':
    main()
