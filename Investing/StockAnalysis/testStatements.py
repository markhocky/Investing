import unittest
import pandas
import warnings
import numpy as np
from mock import Mock
from StockAnalysis.Analysers import Statement, IncomeStatement, BalanceSheet, CashflowStatement

class Test_Statements(unittest.TestCase):

    def setUp(self):
        data_folder = "..\\testData\\"
        self.asset_sheet = pandas.read_pickle(data_folder + "GNG\\GNGassets.pkl")
        self.liabilities = pandas.read_pickle(data_folder + "GNG\\GNGliabilities.pkl")
        self.years = ["2015", "2014", "2013", "2012", "2011"]

        self.zero_series = pandas.Series([0.0, 0.0, 0.0, 0.0, 0.0], index = self.years)

        self.statement = Statement()


    def test_ConvertToFloat(self):
        self.assertEqual(self.statement.convert_to_float("9,000"), 9000.0)
        self.assertEqual(self.statement.convert_to_float("(9,000.0)"), -9000.0)
        self.assertEqual(self.statement.convert_to_float("-"), 0.0)
        self.assertTrue(np.isnan(self.statement.convert_to_float(np.nan)))
        
    def test_StatementReturnsZeroOnMissingKey(self):
        label = "NON-EXISTENT LABEL"
        expected_warning = label + " not found."

        with warnings.catch_warnings(record=True) as w:
            result = self.statement.get_row(self.asset_sheet, label)
            self.assertTrue(result.equals(self.zero_series))
            self.assertRegexpMatches(str(w[-1].message), expected_warning)

    def test_StatementFindsUnits(self):
        self.assertEqual(self.statement.get_units(self.asset_sheet), 1000)

    def test_GetRowAppliesUnitConversion(self):
        self.statement.units = 1000000
        line_item = "Cash Only"
        expected = pandas.Series([64583.0, 33542.3, 16218.7, 33861.2, 36014.1], index = self.years)
        expected = expected * (self.statement.units / 1000)
        actual = self.statement.get_row(self.asset_sheet, line_item)
        self.assertTrue(actual.equals(expected))


class Test_BalanceSheet(unittest.TestCase):
    
    def setUp(self):
        data_folder = "..\\testData\\"
        self.asset_sheet = pandas.read_pickle(data_folder + "GNG\\Financials\\GNGassets.pkl")
        self.liabilities = pandas.read_pickle(data_folder + "GNG\\Financials\\GNGliabilities.pkl")
        years = ["2015", "2014", "2013", "2012", "2011"]

        self.PPE = pandas.Series([3514.6, 2040.9, 2672.0, 2191.9, 1972.6], index = years)
        self.intangibles = pandas.Series([552.7, 3647.7, 0.0, 0.0, 0.0], index = years)
        self.assets = pandas.Series([102972.0, 81903.0, 63604.2, 64117.2, 68824.9], index = years)
        self.cash = pandas.Series([64583.0, 33542.3, 16218.7, 33861.2, 36014.1], index = years)
        self.debt = pandas.Series([1104.3, 535.4, 908.3, 479.0, 928.5], index = years)
        self.other_intangibles = pandas.Series([0.0, 0.0, 0.0, 0.0, 0.0], index = years)

        self.balance = BalanceSheet(self.asset_sheet, self.liabilities)


    def test_BalanceSheetAssets(self):
        self.assertTrue(self.PPE.equals(self.balance.PPE))
        self.assertTrue(self.intangibles.equals(self.balance.intangibles))
        self.assertTrue(self.cash.equals(self.balance.cash))
        self.assertTrue(self.assets.equals(self.balance.assets))
        self.assertTrue(self.other_intangibles.equals(self.balance.other_intangibles))
        

    def test_BalanceSheetLiabilities(self):
        self.assertTrue(self.debt.equals(self.balance.debt))


class Test_IncomeStatement(unittest.TestCase):

    def setUp(self):
        data_folder = "..\\testData\\"
        self.income_sheet = pandas.read_pickle(data_folder + "GNG\\Financials\\GNGincome.pkl")
        self.financial_income = pandas.read_pickle(data_folder + "CCP\\CCPincome.pkl")
        years = ["2015", "2014", "2013", "2012", "2011"]

        self.sales = pandas.Series([216892.6, 114182.9, 114695.4, 152837.9, 142511.6], index = years)
        self.DandA = pandas.Series([4169.4, 1639.2, 974.8, 685.7, 542.4], index = years)
        self.depreciation = pandas.Series([4169.4, 0.0, 974.8, 685.7, 542.4], index = years)
        self.amortization = pandas.Series([0.0] * len(years), index = years)
        interest_earned = pandas.Series([1117.3, 1264.7, 1491.0, 2191.8, 955.8], index = years)
        interest_expense = pandas.Series([58.9, 81.0, 104.0, 88.1, 44.0], index = years)
        self.net_interest = interest_earned - interest_expense
        self.pretax = pandas.Series([17195.9, 16786.6, 11476.0, 19858.1, 29247.5], index = years)
        self.num_shares_diluted = pandas.Series([153645.9, 152880.6, 151569.3, 150000.0, 125932.1], index = years)
        self.COGS = pandas.Series([188919.4, 92752.5, 95962.6, 127909.8, 105753.4], index = years)
        self.SGA = pandas.Series([8135.10, 7170.40, 7966.20, 6447.00, 8320.20], index = years)
        self.SGA_financials = pandas.Series([22004.0, 12395.0, 11372.0, 17302.0, 13957.0], index = years)
        self.net_to_common = pandas.Series([12937.7, 14163.6, 7539.5, 13115.5, 21097.9], index = years)
        self.diluted_EPS = (self.net_to_common / self.num_shares_diluted).round(3)

        self.income = IncomeStatement(self.income_sheet)
        self.financials = IncomeStatement(self.financial_income)


    def test_IncomeSheetValues(self):
        self.assertTrue(self.sales.equals(self.income.sales))
        self.assertTrue(self.DandA.equals(self.income.DandA))
        self.assertTrue(self.depreciation.equals(self.income.depreciation))
        self.assertTrue(self.amortization.equals(self.income.amortization))
        self.assertTrue(self.net_interest.equals(self.income.net_interest))
        self.assertTrue(self.pretax.equals(self.income.pretax))
        self.assertTrue(self.num_shares_diluted.equals(self.income.num_shares_diluted))
        self.assertTrue(self.COGS.equals(self.income.COGS))
        self.assertTrue(self.SGA.equals(self.income.SGA))
        self.assertTrue(self.net_to_common.equals(self.income.net_to_common))
        self.assertTrue(self.diluted_EPS.equals(self.income.diluted_EPS))

    def test_IncomeSheetAliases(self):
        self.assertTrue(self.SGA_financials.equals(self.financials.SGA))


class Test_CashflowStatement(unittest.TestCase):

    def setUp(self):
        data_folder = "..\\testData\\"
        self.operating = pandas.read_pickle(data_folder + "GNG\\Financials\\GNGoperating.pkl")
        self.investing = pandas.read_pickle(data_folder + "GNG\\Financials\\GNGinvesting.pkl")
        self.financing = pandas.read_pickle(data_folder + "GNG\\Financials\\GNGfinancing.pkl")
        years = ["2015", "2014", "2013", "2012", "2011"]

        self.capex_assets = pandas.Series([1797.3, 43.9, 724.1, 908.3, 1081.1], index = years)
        self.capex_other = pandas.Series([0.0, 0.0, 0.0, 0.0, 0.0], index = years)
        self.dividends = pandas.Series([12784.7, 9000.0, 9000.0, 12000.0, 19000.0], index = years)
        self.asset_sales = pandas.Series([0.0, 0.0, 0.0, 0.0, 0.0], index = years)
        self.debt_reduction = pandas.Series([168.5, 358.1, 301.4, 449.5, -277.2], index = years)

        self.cashflow = CashflowStatement(self.operating, self.investing, self.financing)


    def test_CashflowValues(self):
        self.assertTrue(self.capex_assets.equals(self.cashflow.capex_assets))
        print(self.cashflow.capex_other)
        self.assertTrue(self.capex_other.equals(self.cashflow.capex_other))
        self.assertTrue(self.dividends.equals(self.cashflow.dividends_total))
        self.assertTrue(self.asset_sales.equals(self.cashflow.asset_sales))
        self.assertTrue(self.debt_reduction.equals(self.cashflow.debt_reduction))


if __name__ == '__main__':
    unittest.main()
