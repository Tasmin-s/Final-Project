import unittest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.analysis import (
    summary_statistics,
    correlation_analysis,
    trends_over_time,
    regression_analysis,
    save_results_to_csv,
)
import os

class TestAnalysisFunctions(unittest.TestCase):

    def test_summary_statistics(self):
        # Test summary_statistics function
        stats = summary_statistics(self.df, ['aggregated_infant_mortality'])
        self.assertIsInstance(stats, pd.DataFrame)
        self.assertIn('mean', stats.columns)
        self.assertIn('variance', stats.columns)
        expected_mean = self.df['aggregated_infant_mortality'].mean()
        self.assertEqual(stats.at['aggregated_infant_mortality', 'mean'], expected_mean)

    def test_correlation_analysis(self):
        # Test correlation_analysis function
        correlation = correlation_analysis(
            self.df, 'aggregated_infant_mortality', 'aggregated_life_expectancy'
        )
        self.assertIsInstance(correlation, float)
        expected_correlation = self.df[['aggregated_infant_mortality', 'aggregated_life_expectancy']].corr().iloc[0,1]
        self.assertAlmostEqual(correlation, expected_correlation)

    def test_trends_over_time(self):
        # Test trends_over_time function
        trends = trends_over_time(
            self.df, 'aggregated_infant_mortality', 'aggregated_life_expectancy'
        )
        self.assertIsInstance(trends, pd.DataFrame)
        self.assertIn('mortality_rate_change', trends.columns)
        self.assertIn('life_expectancy_change', trends.columns)
        self.assertEqual(len(trends), self.df['year'].nunique())
        # Check that percentage changes are computed correctly
        expected_mortality_change = self.df.groupby('year')['aggregated_infant_mortality'].mean().pct_change()
        pd.testing.assert_series_equal(
            trends['mortality_rate_change'], expected_mortality_change, check_names=False
        )

    def test_regression_analysis(self):
        # Test regression_analysis function
        regression_result = regression_analysis(
            self.df,
            'aggregated_infant_mortality',
            ['year', 'aggregated_life_expectancy']
        )
        self.assertIsInstance(regression_result, sm.iolib.summary.Summary)

    def test_save_results_to_csv(self):
        # Test save_results_to_csv function
        test_file_path = 'test_output.csv'
        save_results_to_csv(self.df, test_file_path)
        self.assertTrue(os.path.exists(test_file_path))
        # Load the file and compare
        loaded_df = pd.read_csv(test_file_path)
        pd.testing.assert_frame_equal(self.df.reset_index(drop=True), loaded_df)
        # Clean up
        os.remove(test_file_path)

if __name__ == '__main__':
    unittest.main()
