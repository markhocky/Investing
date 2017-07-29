import unittest
import openpyxl
from DataHandling.Downloads import XLSio, Storage
from pandas import DataFrame
from mock import Mock


class Test_XLSioReadingWorksheets(unittest.TestCase):

    def setUp(self):
        self.existing_tickers = ["1PG", "ONT", "1ST", "TGP", "TIX"]
        self.existing_headings = ["Company name", "ASX code", "GICS industry group"]
        self.xls = XLSio(Storage("..\\testData\\"))
        self.xls.loadWorkbook("ASXListedCompanies")


    def test_ExtractTickersList(self):
        tickers = self.xls.getTickers()
        self.assertEqual(len(tickers), 5)
        self.assertEqual(tickers[0:len(self.existing_tickers)], self.existing_tickers)

    def test_GetHeader(self):
        expected_row = ["Company name", "GICS industry group"]
        header_row = self.xls.getHeader()
        self.assertEqual(header_row, expected_row)

    def test_StockTable(self):
        expected_headings = ["Company name", "GICS industry group"]
        stock_table = self.xls.table
        self.assertIsInstance(stock_table, DataFrame)
        self.assertSetEqual(set(stock_table.index), set(self.existing_tickers))
        self.assertSetEqual(set(stock_table.loc["1PG"].index), set(expected_headings))
        self.assertDictEqual(dict(stock_table.loc["TGP"]), dict(zip(expected_headings, ["360 CAPITAL GROUP", "Real Estate"])))


class Test_XLSisUpdatingWorksheets(unittest.TestCase):

    def setUp(self):
        self.existing_tickers = ["1PG", "ONT", "1ST", "TGP", "TIX"]
        self.existing_headings = ["Company name", "GICS industry group"]
        self.xls = XLSio(Storage("..\\testData\\"))
        self.xls.loadWorkbook("ASXListedCompanies")


    def test_AddNewData(self):
        tickers = ["ONT", "TGP"]
        new_entries = [{"Market Cap" : 450, "EPS" : 0.30}, 
                       {"Market Cap" : 5, "EPS" : "N/A"}]
        new_data = dict(zip(tickers, new_entries))
        new_table = DataFrame.from_dict(new_data, orient = "index")
        existing_table = self.xls.table
        expected_table = existing_table.join(new_table)

        self.xls.updateTable(new_data)
        stock_table = self.xls.table
        self.assertListEqual(list(stock_table.index), list(expected_table.index))
        self.assertDictEqual(dict(stock_table.loc["ONT"]), dict(expected_table.loc["ONT"]))


class Test_XLSioSavingWorkbooks(unittest.TestCase):

    def setUp(self):
        self.filename = "test_file"
        store = Storage("..\\testData\\")
        self.filepath = store.excel("test_file")
        self.xls = XLSio(store)
        self.xls.table = Mock()
        self.xls.table.to_excel = Mock()

    def test_SavingWorkbookAsNewFile(self):
        self.xls.saveAs(self.filename)
        self.xls.table.to_excel.assert_called_once_with(self.filepath, sheet_name = "Stock table")
        
        

if __name__ == '__main__':
    unittest.main()
