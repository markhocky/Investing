import unittest
from DataHandling.Downloads import WSJdownloader, Storage

stock_overview = "http://quotes.wsj.com/AU/XASX/[ticker]"
financials_overview = "http://quotes.wsj.com/AU/XASX/[ticker]/financials"
annual_income_statement = "http://quotes.wsj.com/AU/XASX/[ticker]/financials/annual/income-statement"
annual_balance_sheet = "http://quotes.wsj.com/AU/XASX/[ticker]/financials/annual/balance-sheet"
annual_cash_flow = "http://quotes.wsj.com/AU/XASX/[ticker]/financials/annual/cash-flow"

tickers = ["SND", "SRV", "BHP"]

class Test_BuildingQueryAddress(unittest.TestCase):
    
    def test_BuildsStockOverviewAddress(self):
        scraper = WSJdownloader(Storage(""))
        for ticker in tickers:
            scraper_address = scraper.get_address(ticker, "overview")
            expected_address = stock_overview.replace("[ticker]", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsFinancialsOverviewAddress(self):
        scraper = WSJdownloader(Storage(""))
        for ticker in tickers:
            scraper_address = scraper.get_address(ticker, "financials")
            expected_address = financials_overview.replace("[ticker]", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsIncomeStatementAddress(self):
        scraper = WSJdownloader(Storage(""))
        for ticker in tickers:
            scraper_address = scraper.get_address(ticker, "income")
            expected_address = annual_income_statement.replace("[ticker]", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsBalanceSheetAddress(self):
        scraper = WSJdownloader(Storage(""))
        for ticker in tickers:
            scraper_address = scraper.get_address(ticker, "balance")
            expected_address = annual_balance_sheet.replace("[ticker]", ticker)
            self.assertEqual(scraper_address, expected_address)

    def test_BuildsCashFlowAddress(self):
        scraper = WSJdownloader(Storage(""))
        for ticker in tickers:
            scraper_address = scraper.get_address(ticker, "cashflow")
            expected_address = annual_cash_flow.replace("[ticker]", ticker)
            self.assertEqual(scraper_address, expected_address)



if __name__ == '__main__':
    unittest.main()
