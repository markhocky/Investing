import unittest
import pandas
from mock import Mock
from StockAnalysis.Analysers import Reporter, EPVanalyser

class Test_Reporter(unittest.TestCase):

    def setUp(self):
        self.ticker = "GNG"
        self.years = ["2015", "2014", "2013", "2012", "2011"]
        self.reporter = Reporter(self.ticker)
        self.reporter.index = self.years

    def test_ReporterBuildsHeaderRow(self):
        expected = pandas.Series(["*****"] * len(self.years), index = self.years)
        expected.name = "Heading"
        self.assertTrue(expected.equals(self.reporter.heading("Heading")))

    def test_PadRow(self):
        price = 0.70
        expected = pandas.Series([price] + ["-"] * (len(self.years) - 1), index = self.years)
        row = self.reporter.pad_row(price)
        self.assertTrue(row.equals(expected))


class Test_EPVreporting(unittest.TestCase):

    def setUp(self):
        self.analyser = Mock(spec = EPVanalyser)
        self.analyser.EPV_base = Mock(return_value = 0)
        self.analyser.EPV_minimum = Mock(return_value = 0)
        self.analyser.EPV_maximum = Mock(return_value = 0)
        self.analyser.EPV_adjusted = Mock(return_value = 0)
        self.analyser.EPV_levered = Mock(return_value = 0)
        self.analyser.EPV_growth = Mock(return_value = 0)
        self.analyser.EPV_cyclic = Mock(return_value = 0)
        self.analyser.EPV_diluted = Mock(return_value = 0)

        self.reporter = Reporter("test", analyse = False)
        self.reporter.EPVanalyser = self.analyser

    def test_EPVtable(self):
        table = self.reporter.EPV_table()
        expected_headings = ["Adjusted", "Min", "Max", "Base", "Levered", "Growth", "Cyclic", "Dilution"]
        self.assertEqual(table.columns.tolist(), expected_headings)

        self.assertTrue(self.analyser.EPV_base.called)
        self.assertTrue(self.analyser.EPV_minimum.called)
        self.assertTrue(self.analyser.EPV_maximum.called)
        self.assertTrue(self.analyser.EPV_levered.called)
        self.assertTrue(self.analyser.EPV_adjusted.called)
        self.assertTrue(self.analyser.EPV_growth.called)
        self.assertTrue(self.analyser.EPV_cyclic.called)
        self.assertTrue(self.analyser.EPV_diluted.called)

    

if __name__ == '__main__':
    unittest.main()
