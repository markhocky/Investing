import unittest
from mock import Mock, call
from DataHandling.Downloads import WebDownloader, WSJinternet, Storage, Financials

stock_overview = "http://quotes.wsj.com/AU/XASX/<ticker>"
financials_overview = "http://quotes.wsj.com/AU/XASX/<ticker>/financials"
annual_income_statement = "http://quotes.wsj.com/AU/XASX/<ticker>/financials/annual/income-statement"
quarter_income_statement = "http://quotes.wsj.com/AU/XASX/<ticker>/financials/quarter/income-statement"
annual_balance_sheet = "http://quotes.wsj.com/AU/XASX/<ticker>/financials/annual/balance-sheet"
annual_cash_flow = "http://quotes.wsj.com/AU/XASX/<ticker>/financials/annual/cash-flow"


class Test_WebDownloader(unittest.TestCase):
    '''
    Responsibility of WebDownloader is to act as a middle man between web resource and storage.
    WebDownloader handles which tickers are to be updated and selects the appropriate web resource
    for the request.
    '''

    def setUp(self):
        self.downloader = WebDownloader()
        self.tickers = ["ASX", "BHP", "CBA"]


    def test_UpdateFinancials(self):
        period = "annual"
        existing_financials = Mock()
        existing_financials.merge = Mock()
        new_financials = Mock()
        self.downloader.WSJ.getFinancials = Mock(return_value = new_financials)
        self.downloader.store.loadFinancials = Mock(return_value = existing_financials)
        self.downloader.store.saveFinancials = Mock()

        self.downloader.updateFinancials(self.tickers, period)

        expected_WSJ_calls = [call(ticker, period) for ticker in self.tickers]
        self.downloader.WSJ.getFinancials.assert_has_calls(expected_WSJ_calls)

        expected_store_loads = [call(ticker, period) for ticker in self.tickers]
        self.downloader.store.loadFinancials.assert_has_calls(expected_store_loads)

        expected_financial_merges = [call(new_financials)] * len(self.tickers)
        existing_financials.merge.assert_has_calls(expected_financial_merges)

        expected_store_saves = [call(ticker, existing_financials,  period) for ticker in self.tickers]
        self.downloader.store.saveFinancials.assert_has_calls(expected_store_saves)


    def test_GetCMCsummaries(self):
        pass

    def test_GetPriceHistories(self):
        pass

    def test_GetLatestPrices(self):
        pass



class Test_FinancialsHolder(unittest.TestCase):

    def setUp(self):
        self.financials = Financials("ASX", "annual")
        self.new_financials = Financials("ASX", "annual")
        for sheet in ["income", "balance", "cashflow"]:
            self.new_financials.statements[sheet] = sheet

    def test_FinancialsChecksTickerAndPeriodBeforeMerge(self):
        new_fin1 = Financials("BHP", "annual")
        new_fin2 = Financials("ASX", "interim")
        new_fin3 = Financials("BHP", "interim")

        self.assertRaises(ValueError, self.financials.merge, new_fin1)
        self.assertRaises(ValueError, self.financials.merge, new_fin2)
        self.assertRaises(ValueError, self.financials.merge, new_fin3)

    def test_FinancialsAssignsNewIfCurrentlyEmpty(self):

        self.financials.merge(self.new_financials)

        for sheet in self.financials.statements:
            self.assertEqual(self.financials.statements[sheet], self.new_financials.statements[sheet])

    def test_NoChangeToFinancialsIfNewIsEmpty(self):
        self.financials.statements = {"income" : Mock(), "balance" : Mock(), "cashflow" : Mock()}
        for sheet in self.financials.statements:
            self.financials.statements[sheet].merge = Mock()

        self.new_financials.statements = {}

        self.financials.merge(self.new_financials)

        for sheet in self.financials.statements:
            self.financials.statements[sheet].merge.assert_not_called()

    def test_FinancialsAsDict(self):

        fin_dict = self.financials.toDict()

        self.assertEqual(fin_dict["ticker"], self.financials.ticker)
        self.assertEqual(fin_dict["period"], self.financials.period)
        self.assertEqual(fin_dict["statements"], self.financials.statements)

    def test_FinancialsCallsAppropriateFolder(self):
        store = Storage()
        store.financials = Mock()
        self.financials.selectFolder(store)
        store.financials.assert_called_once_with(self.financials)

    def test_FinancialsReturnsFilename(self):
        expected_filename = self.financials.ticker + self.financials.period + ".pkl"
        self.assertEqual(self.financials.filename(), expected_filename)



class Test_WSJresource(unittest.TestCase):

    def setUp(self):
        self.WSJ = WSJinternet()
        self.mock_page = "<html>"
        self.WSJ.load_page = Mock(return_value = self.mock_page)
        self.WSJ.scraper.getTables = Mock(return_value = None)


    def test_ResourceReturnsFinancialsObject(self):
        self.assertIsInstance(self.WSJ.getFinancials("ASX", "annual"), Financials)

    def test_ResourceGetsAllStatementTables(self):
        self.WSJ.getFinancials("ASX", "annual")

        expected_scraper_calls = [call(sheet, self.mock_page) for sheet in ["income", "balance", "cashflow"]]

        self.WSJ.scraper.getTables.assert_has_calls(expected_scraper_calls, any_order = True)



class Test_BuildingQueryAddress(unittest.TestCase):

    def setUp(self):
        self.scraper = WSJinternet()
        self.tickers = ["SND", "SRV", "BHP"]

    
    def test_BuildsStockOverviewAddress(self):
        for ticker in self.tickers:
            scraper_address = self.scraper.get_address(ticker, "overview")
            expected_address = stock_overview.replace("<ticker>", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsFinancialsOverviewAddress(self):
        for ticker in self.tickers:
            scraper_address = self.scraper.get_address(ticker, "financials")
            expected_address = financials_overview.replace("<ticker>", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsIncomeStatementAddress(self):
        for ticker in self.tickers:
            scraper_address = self.scraper.get_address(ticker, "income")
            expected_address = annual_income_statement.replace("<ticker>", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsBalanceSheetAddress(self):
        for ticker in self.tickers:
            scraper_address = self.scraper.get_address(ticker, "balance")
            expected_address = annual_balance_sheet.replace("<ticker>", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsCashFlowAddress(self):
        for ticker in self.tickers:
            scraper_address = self.scraper.get_address(ticker, "cashflow")
            expected_address = annual_cash_flow.replace("<ticker>", ticker)
            self.assertEqual(scraper_address, expected_address)


class Test_BuildHYAddress(unittest.TestCase):

    def setUp(self):
        self.scraper = WSJinternet(Storage())

    def test_ThrowsErrorIfPeriodMisspecified(self):
        self.assertRaises(ValueError, self.scraper.get_address, "ASX", "balance", "x")
        self.assertRaisesRegexp(ValueError, "Should be 'annual' or 'quarter'", 
                                self.scraper.get_address, "ASX", "balance", "x")

    def test_BuildsHYIncomeStatmentAddress(self):
        ticker = "ASX"
        scraper_address = self.scraper.get_address(ticker, "income", "quarter")
        expected_address = quarter_income_statement.replace("<ticker>", ticker)
        self.assertEqual(scraper_address, expected_address)


if __name__ == '__main__':
    unittest.main()