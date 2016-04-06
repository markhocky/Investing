import unittest
from DataHandling.Downloads import Storage

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



if __name__ == '__main__':
    unittest.main()
