import unittest
from bs4 import BeautifulSoup
from DataHandling.Downloads import WSJscraper, Storage, InsufficientDataError
import pandas


class Test_LoadingPages(unittest.TestCase):

    def setUp(self):
        self.ticker = "SRV"
        self.type = "overview"
        self.data_dir = "..\\testData\\"
        self.store = Storage(self.data_dir)
        self.scraper = WSJscraper(self.store)


    def test_StoresSoupObjectOnPageLoad(self):
        self.scraper.load_overview(self.ticker)
        expected = BeautifulSoup(open(self.store.html(self.ticker, "overview")), "lxml")
        self.assertIsInstance(self.scraper.overview, BeautifulSoup)
        self.assertEqual(self.scraper.overview.title, expected.title)

    def test_ChecksForValidYears(self):
        years = [u'2015', u' ', u'012', u'.3']
        self.assertRaises(InsufficientDataError, self.scraper.check_years, years = years)
        self.assertIsNone(self.scraper.check_years([u'2015', u'2014', u'2013']))


class Test_ScrapingOverviewData(unittest.TestCase):

    def setUp(self):
        self.ticker = "SRV"
        self.type = "overview"
        self.data_dir = "..\\testData\\"
        self.store = Storage(self.data_dir)
        self.expected = {"P/E Ratio (TTM)" : 21.68, 
                         "EPS (TTM)" : 0.34, 
                         "Market Cap" : 698.87, 
                         "Shares Outstanding" : 98.43, 
                         "Public Float" : 43.03, 
                         "Yield" : 3.01, 
                         "Latest Dividend" : 0.11, 
                         "Ex-Dividend Date" : "09/07/15"}
        self.scraper = WSJscraper(self.store)
        self.scraper.overview = BeautifulSoup(open(self.store.html(self.ticker, self.type)), "lxml")


    def test_RetrievesKeyStockDataTable(self):
        data = self.scraper.keyStockData()
        # scraper should return a dictionary
        self.assertIsInstance(data, dict)
        self.assertEqual(data.keys().sort(), self.expected.keys().sort())
        self.assertEqual(data, self.expected)

class Test_ScrapingBalanceSheet(unittest.TestCase):

    def setUp(self):
        self.ticker = "GNG"
        self.type = "balance"
        data_dir = "..\\testData\\"
        store = Storage(data_dir)
        self.assets = pandas.read_pickle(data_dir + "GNG\\GNGassets.pkl")
        self.liabilities = pandas.read_pickle(data_dir + "GNG\\GNGliabilities.pkl")
        self.scraper = WSJscraper(store)


    def test_ReadAssetsToDataFrame(self):
        table = self.scraper.read_statement_table(self.ticker, self.type, contains = "Cash Only")
        self.assertIsInstance(table, pandas.DataFrame)
        self.assertEqual(table.shape[1], 5)
        self.assertTrue(table.equals(self.assets))

    def test_ReadLiabilitiesToDataFrame(self):
        table = self.scraper.read_statement_table(self.ticker, self.type, contains = "Short Term Debt")
        self.assertIsInstance(table, pandas.DataFrame)
        self.assertEqual(table.shape[1], 5)
        self.assertTrue(table.equals(self.liabilities))



if __name__ == '__main__':
    unittest.main()
