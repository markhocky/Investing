import unittest
from DataHandling.Downloads import WSJdownloader, Storage

stock_overview = "http://quotes.wsj.com/AU/XASX/[ticker]"
financials_overview = "http://quotes.wsj.com/AU/XASX/[ticker]/financials"
annual_income_statement = "http://quotes.wsj.com/AU/XASX/[ticker]/financials/annual/income-statement"
quarter_income_statement = "http://quotes.wsj.com/AU/XASX/[ticker]/financials/quarter/income-statement"
annual_balance_sheet = "http://quotes.wsj.com/AU/XASX/[ticker]/financials/annual/balance-sheet"
annual_cash_flow = "http://quotes.wsj.com/AU/XASX/[ticker]/financials/annual/cash-flow"


class Test_BuildingQueryAddress(unittest.TestCase):

    def setUp(self):
        self.scraper = WSJdownloader(Storage())
        self.tickers = ["SND", "SRV", "BHP"]

    
    def test_BuildsStockOverviewAddress(self):
        for ticker in self.tickers:
            scraper_address = self.scraper.get_address(ticker, "overview")
            expected_address = stock_overview.replace("[ticker]", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsFinancialsOverviewAddress(self):
        for ticker in self.tickers:
            scraper_address = self.scraper.get_address(ticker, "financials")
            expected_address = financials_overview.replace("[ticker]", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsIncomeStatementAddress(self):
        for ticker in self.tickers:
            scraper_address = self.scraper.get_address(ticker, "income")
            expected_address = annual_income_statement.replace("[ticker]", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsBalanceSheetAddress(self):
        for ticker in self.tickers:
            scraper_address = self.scraper.get_address(ticker, "balance")
            expected_address = annual_balance_sheet.replace("[ticker]", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsCashFlowAddress(self):
        for ticker in self.tickers:
            scraper_address = self.scraper.get_address(ticker, "cashflow")
            expected_address = annual_cash_flow.replace("[ticker]", ticker)
            self.assertEqual(scraper_address, expected_address)


class Test_BuildHYAddress(unittest.TestCase):

    def setUp(self):
        self.scraper = WSJdownloader(Storage())

    def test_ThrowsErrorIfPeriodMisspecified(self):
        self.assertRaises(ValueError, self.scraper.get_address, "ASX", "balance", "x")
        self.assertRaisesRegexp(ValueError, "Should be 'annual' or 'quarter'", 
                                self.scraper.get_address, "ASX", "balance", "x")

    def test_BuildsHYIncomeStatmentAddress(self):
        ticker = "ASX"
        scraper_address = self.scraper.get_address(ticker, "income", "quarter")
        expected_address = quarter_income_statement.replace("[ticker]", ticker)
        self.assertEqual(scraper_address, expected_address)


if __name__ == '__main__':
    unittest.main()