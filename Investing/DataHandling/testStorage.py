import unittest
import os
from DataHandling.Downloads import Storage
import shutil

class Test_Storage(unittest.TestCase):

    def setUp(self):
        self.root_location = "..\\testData\\"
        self.store = Storage(self.root_location)


    def test_StorageLocationForExcel(self):
        filename = "test"
        expected_filepath = self.root_location + filename + ".xlsx"
        self.assertEqual(self.store.excel(filename), expected_filepath)

    def test_StorageLocationForHTML(self):
        ticker = "AAA"
        type = "overview"
        expected_filepath = self.root_location + ticker + "\\" + ticker + type + ".html"
        self.assertEqual(self.store.html(ticker, type), expected_filepath)

    def test_StorageLocationForStockPickle(self):
        ticker = "AAA"
        type = "income"
        expected_filepath = self.root_location + ticker + "\\pickles\\" + type + ".pkl"
        self.assertEqual(self.store.stock_pickle(ticker, type), expected_filepath)

    def test_StorageLocationForFinancials(self):
        ticker = "AAA"
        type = "income"
        expected_annual_folder = self.root_location + ticker + "\\Financials\\Annual"
        expected_interim_folder = self.root_location + ticker + "\\Financials\\Interim"
        self.assertEqual(self.store.get_folder(ticker, "annual financials"), expected_annual_folder)
        self.assertEqual(self.store.get_folder(ticker, "interim financials"), expected_interim_folder)




    #def test_StorageMigratesAllFilesInFolder(self):

    #    old_folder = "..\\testData\\GNG\\pickle_move"
    #    new_root = "..\\testData"
    #    store = Storage(root_folder = new_root)
    #    expected_file = store.stock_pickle("GNG", "GNGassets")
    #    expected_folder = os.path.dirname(expected_file)
        
    #    store.migrate(old_folder, "stock_pickle", "GNG")

    #    self.assertTrue(os.path.exists(expected_file))
    #    self.assertTrue(os.path.exists(expected_folder))
    #    self.assertFalse(os.path.exists(old_folder))

    #def test_StorageMigratesFilesMatchingPattern(self):
    #    existing_folder = "..\\testData\\GNG"
    #    existing_yahoo_file = "GNGprices.pkl"
    #    existing_html_file = "GNGbalance.html"
    #    new_root = "..\\testData\\GNGmove"
    #    expected_file = os.path.join(new_root, "GNG\\GNGprices.pkl")
    #    storage = Storage(root_folder = new_root)
        
    #    storage.migrate(existing_folder, "yahoo", "GNG", "prices.pkl")

    #    self.assertTrue(os.path.exists(os.path.join(existing_folder, existing_html_file)))
    #    self.assertTrue(os.path.exists(expected_file))
    #    self.assertFalse(os.path.exists(os.path.join(existing_folder, existing_yahoo_file)))


if __name__ == '__main__':
    unittest.main()
