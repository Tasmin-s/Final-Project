import unittest
import pandas as pd
from scipy.stats import pearsonr
from src.analysis import summary_statistics, correlation_analysis_with_test
from src.data_processing import load_data

# Define the path to the merged data
merged_data = './data/processed/merged_data.csv'

def load_merged_data():
    """
    Load the merged dataset for testing.
    """
    return load_data(merged_data)

class TestAnalysisFunctions(unittest.TestCase):
    #  test summary statistice 
    def test_summary_statistics(self):
        """
        Test the summary_statistics function using the actual merged dataset.
        """
        df = load_merged_data()
        stats = summary_statistics(df, ['aggregated_infant_mortality'])
        self.assertIsInstance(stats, pd.DataFrame)
        self.assertIn('mean', stats.columns)
        self.assertIn('variance', stats.columns)
        expected_mean = df['aggregated_infant_mortality'].mean()
        self.assertAlmostEqual(stats.at['aggregated_infant_mortality', 'mean'], expected_mean)
    # test correlation analysis
    def test_correlation_analysis_with_test(self):
        """
        Test the correlation analysis with hypothesis testing using the actual merged dataset.
        """
        df = load_merged_data()
        correlation, p_value, result = correlation_analysis_with_test(
            df,
            'aggregated_infant_mortality',
            'aggregated_life_expectancy'
        )

        # Check if the correlation is a float
        self.assertIsInstance(correlation, float)

        # Check if the p-value is a float
        self.assertIsInstance(p_value, float)

        # Check if result is a string
        self.assertIsInstance(result, str)

        # Verify correlation matches manual calculation
        expected_correlation, expected_p_value = pearsonr(
            df['aggregated_infant_mortality'],
            df['aggregated_life_expectancy']
        )
        self.assertAlmostEqual(correlation, expected_correlation)
        self.assertAlmostEqual(p_value, expected_p_value)

if __name__ == '__main__':
    unittest.main()

