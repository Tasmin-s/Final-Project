import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages

# create summary statistics
def summary_statistics(df, columns):
    """
    Calculate summary statistics for specified columns.
    """
    summary = df[columns].describe().transpose()
    summary['variance'] = df[columns].var()
    return summary

# carry out hypothesis test 
def correlation_analysis_with_test(df, mortality_column, life_expectancy_column, alpha=0.05):
    """
    Compute the correlation between infant mortality and life expectancy
    and perform a hypothesis test.
    
    H0: There is no correlation between infant mortality and life expectancy.
    H1: There is a significant correlation between infant mortality and life expectancy.
    """
    correlation, p_value = pearsonr(df[mortality_column], df[life_expectancy_column])
    
    if p_value < alpha:
        result = "Reject the null hypothesis (H0): Significant correlation exists."
    else:
        result = "Fail to reject the null hypothesis (H0): No significant correlation."

    return correlation, p_value, result

def save_summary_to_pdf(summary_df, output_path, caption):
    """
    Save summary statistics to a PDF with a table and caption.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.round(2).values,
                     colLabels=summary_df.columns,
                     rowLabels=summary_df.index,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.figtext(0.5, 0.02, caption, wrap=True, horizontalalignment='center', fontsize=12)
    
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    merged_data_path = '/Users/Tasmin/Final-Project/data/processed/merged_data.csv'
    pdf_output_path = '/Users/Tasmin/Final-Project/data/processed/summary_statistics.pdf'
    
    print("Loading Processed Data...")
    merged_data = pd.read_csv(merged_data_path)
    
    aggregated_infant_mortality_column = 'aggregated_infant_mortality'
    aggregated_life_expectancy_column = 'aggregated_life_expectancy'
    
    print("Calculating Summary Statistics...")
    summary_stats = summary_statistics(merged_data, [aggregated_infant_mortality_column, aggregated_life_expectancy_column])
    print(summary_stats)
    
    caption_text = "Table 1: Summary Statistics for Aggregated Infant Mortality and Life Expectancy."
    save_summary_to_pdf(summary_stats, pdf_output_path, caption_text)
    
    print("Performing Correlation Analysis with Hypothesis Test...")
    correlation, p_value, test_result = correlation_analysis_with_test(
        merged_data,
        aggregated_infant_mortality_column,
        aggregated_life_expectancy_column
    )
    
    print(f"Pearson Correlation Coefficient: {correlation}")
    print(f"P-value: {p_value}")
    print(f"Hypothesis Test Result: {test_result}")