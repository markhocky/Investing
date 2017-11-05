import unittest
import datetime
import pandas
from mock import Mock
from StockAnalysis.Analysers import PriceAnalyser

class Test_PriceAnalyser(unittest.TestCase):

    def setUp(self):
        data_folder = "..\\testData\\"
        self.prices = pandas.read_pickle(data_folder + "GNG\\GNGprices.pkl")
        FY_start = datetime.date(2014, 7, 1)
        FY_end = datetime.date(2015, 6, 30)
        self.FY_2015 = self.prices[FY_start:FY_end]
        self.analyser = PriceAnalyser(self.prices)

        
    def test_AnalyserFindsPricesForYear(self):
        self.assertTrue(self.FY_2015.equals(self.analyser.prices_FY(2015)))

    def test_AnalyserFindsPriceRanges(self):
        self.analyser.prices_FY = Mock(return_value = self.FY_2015)
        average = self.FY_2015["Close"].mean()
        highest = self.FY_2015["High"].max()
        lowest = self.FY_2015["Low"].min()
        std_dev = self.FY_2015["Close"].std()
        year = 2015
        self.assertEquals(average, self.analyser.average_price(year))
        self.assertEquals(highest, self.analyser.high_price(year))
        self.assertEquals(lowest, self.analyser.low_price(year))
        self.assertEquals(std_dev, self.analyser.stddev_price(year))

    def test_AnalyserReturnsPriceTable(self):
        years = ["2015", "2014", "2013"]
        price_table = self.analyser.price_table(years)
        self.assertEqual(price_table.columns.tolist(), ["Average", "High", "Low", "Std Dev"])
        self.assertEqual(price_table.index.tolist(), years)
        self.assertAlmostEqual(price_table["Average"]["2015"], 0.813, places = 3)
        self.assertAlmostEqual(price_table["High"]["2015"], 0.980, places = 3)
        self.assertAlmostEqual(price_table["Low"]["2015"], 0.625, places = 3)
        self.assertAlmostEqual(price_table["Std Dev"]["2015"], 0.084, places = 3)



if __name__ == '__main__':
    unittest.main()
