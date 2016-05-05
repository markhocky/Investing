import unittest
import pandas
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from mock import Mock, call
from StockAnalysis.Analysers import EPVanalyser, FinanceAnalyst



class Test_EstimateIncomeAndExpenses(unittest.TestCase):

    def setUp(self):
        years = ["2015", "2014", "2013", "2012", "2011"]
        self.income = Mock()
        self.balance = Mock()
        self.cash = Mock()

        self.zero_series = pandas.Series([0.0] * len(years), index = years)

        self.income.sales = pandas.Series([216892.6, 114182.9, 114695.4, 152837.9, 142511.6], index = years)
        self.income.DandA = pandas.Series([4169.4, 1639.2, 974.8, 685.7, 542.4], index = years)
        self.income.depreciation = pandas.Series([2169.4, 1200.2, 800.8, 650.7, 500.4], index = years)
        self.income.amortization = self.income.DandA - self.income.depreciation
        interest_earned = pandas.Series([1117.3, 1264.7, 1491.0, 2191.8, 955.8], index = years)
        interest_expense = pandas.Series([58.9, 81.0, 104.0, 88.1, 44.0], index = years)
        self.income.net_interest = interest_earned - interest_expense
        self.income.pretax = pandas.Series([17195.9, 16786.6, 11476.0, 19858.1, 29247.5], index = years)
        self.income.unusual_expense = pandas.Series([0.0, 0.0, 0.0, 1000.0, 0.0], index = years)
        self.income.non_operating_income = pandas.Series([4000.0, 17000.0, 4000.0, 18000.0, 0.0], index = years)

        self.balance.assets = pandas.Series([102972.0, 81903.0, 63604.2, 64117.2, 68824.9], index = years)
        self.balance.PPE = pandas.Series([3514.6, 2040.9, 2672.0, 2191.9, 1972.6], index = years)
        self.balance.intangibles = pandas.Series([552.7, 3647.7, 0.0, 0.0, 0.0], index = years)
        self.balance.other_intangibles = pandas.Series([52.70, 3595.0, 0.0, 0.0, 0.0], index = years)
        self.balance.cash = pandas.Series([64583.0, 33542.3, 16218.7, 33861.2, 36014.1], index = years)
        self.balance.debt = pandas.Series([1104.3, 535.4, 908.3, 479.0, 928.5], index = years)
        
        self.cash.capex_assets = pandas.Series([1797.3, 43.9, 724.1, 908.3, 1081.1], index = years)
        self.cash.capex_other = pandas.Series([102.0, 3000.0, 0.0, 0.0, 0.0], index = years)
        self.cash.asset_sales = pandas.Series([0.0, 20.0, 0.0, 0.0, 52.0], index = years)
        self.cash.debt_reduction = pandas.Series([168.0, 358.0, 301.0, 449.5, 277.0], index = years)
        self.cash.change_in_working_capital = pandas.Series([1680.0, -1358.0, 301.0, 4492.5, -2277.0], index = years)
        
        self.lease_financing = pandas.Series([737.4, -15.0, 730.7, 0.0, 0.0], index = years)
        self.adjusted_earnings = pandas.Series([11150.46, 7893.12, 5666.38, 12404.10, 19816.02], index = years)
        self.asset_maintenance = pandas.Series([1850.0, 1658.0, 650.0, 1100.0, 980.0], index = years)

        self.financials = FinanceAnalyst(self.income, self.balance, self.cash)


    def test_AvgCapitalToSales(self):
        avg_capital_sales_ratio = self.financials.cap_sales_ratio()
        yearly_ratios = (self.balance.PPE + self.balance.intangibles) / self.income.sales
        self.assertEqual(avg_capital_sales_ratio, yearly_ratios.mean())
        self.assertAlmostEqual(avg_capital_sales_ratio, 0.024, places = 3)

    def test_EstimateGrowthCapexFromSales(self):
        sales_change = self.income.sales.diff(periods = -1)
        sales_change = sales_change.fillna(method = "pad")
        cap_sales_ratio = 0.05
        expected = sales_change * cap_sales_ratio
        expected[expected < 0] = 0
        self.financials.cap_sales_ratio = Mock(return_value = cap_sales_ratio)
        actual = self.financials.growth_capex()
        self.assertTrue(actual.equals(expected))

    def test_ImpliedCapex(self):
        capital_base_change = (self.balance.PPE + self.balance.intangibles).diff(periods = -1)
        capital_base_change = capital_base_change.fillna(method = "pad")
        expected = capital_base_change + self.income.DandA
        actual = self.financials.implied_capex()
        self.assertTrue(actual.equals(expected))

    def test_ExpendedDepreciation(self):
        capital_base = self.financials.capital_base()
        capital_base_change = self.financials.series_diff(capital_base)
        capital_base_change[capital_base_change > 0] = 0
        expected = self.income.DandA + capital_base_change
        actual = self.financials.expended_depreciation()
        self.assertTrue(actual.equals(expected))

    def test_NetAssetCashExpenditures(self):
        expected = self.cash.capex_assets
        actual = self.financials.asset_capex()
        self.assertTrue(actual.equals(expected))

    def test_LeaseFinancingCosts(self):
        debt_change = self.financials.series_diff(self.financials.totalDebt())
        expected = self.cash.debt_reduction + debt_change
        actual = self.financials.lease_financing()
        self.assertTrue(actual.equals(expected))

    def test_NetPPEMaintenanceCashflows(self):
        self.financials.lease_financing = Mock(return_value = self.lease_financing)
        PPE_change = self.financials.PPE_change_net_sales()
        net_cash_capex = self.financials.asset_capex()
        expected = self.lease_financing + net_cash_capex - PPE_change
        actual = self.financials.PPE_maintenance()
        self.assertTrue(actual.equals(expected))

    def test_PPEmaintenanceWithZeroCashflowError(self):
        self.financials.asset_capex = Mock(return_value = self.zero_series)
        self.financials.lease_financing = Mock(return_value = self.zero_series)
        expected = self.income.depreciation
        actual = self.financials.PPE_maintenance()
        self.assertTrue(actual.equals(expected))

    def test_IntangiblesMaintenanceCashflows(self):
        intangibles_change = self.financials.series_diff(self.balance.other_intangibles)
        intangibles_change[intangibles_change < 0] = 0
        capex_ex_growth = self.cash.capex_other - intangibles_change
        capex_ex_growth[capex_ex_growth < 0] = 0
        intangibles_spend_pct = capex_ex_growth / self.balance.other_intangibles
        expected = self.balance.other_intangibles * intangibles_spend_pct.mean()
        actual = self.financials.intangibles_maintenance()
        self.assertTrue(actual.equals(expected))

    def test_IntangibleMaintenanceWithZeroCashflowError(self):
        self.cash.capex_other = self.zero_series
        expected = self.income.amortization
        actual = self.financials.intangibles_maintenance()
        self.assertTrue(actual.equals(expected))

    def test_InvestedCapital(self):
        net_cash = self.balance.cash - 0.015 * self.income.sales
        expected = self.balance.assets - net_cash
        invested_capital = self.financials.investedCapital()
        self.assertTrue(invested_capital.equals(expected))

    def test_NetUnusuals(self):
        unusuals = self.income.unusual_expense - self.income.non_operating_income
        expected = unusuals - unusuals.mean()
        actual = self.financials.net_unusuals()
        self.assertTrue(actual.equals(expected))

    def test_NonCashExpenses(self):
        expected = self.income.DandA
        actual = self.financials.non_cash_expenses()
        self.assertTrue(actual.equals(expected))

    def test_CashExpenses(self):
        expected = (self.financials.PPE_maintenance() + 
            self.financials.intangibles_maintenance() + self.financials.working_capital_requirements())
        actual = self.financials.cash_expenses()
        self.assertTrue(actual.equals(expected))

    def test_CalculateEBIT(self):
        expected = self.income.pretax - self.income.net_interest
        actual = self.financials.EBIT()
        self.assertTrue(actual.equals(expected))

    def test_CalculateAdjustedEarnings(self):
        self.financials.asset_maintenance = Mock(return_value = self.asset_maintenance)
        EBIT = self.financials.EBIT()
        expected = (EBIT + self.financials.net_unusuals()) * 0.7 + self.financials.non_cash_expenses() - self.asset_maintenance
        adj_earnings = self.financials.ownerEarnings()
        self.assertTrue(adj_earnings.equals(expected))


class Test_EstimatesOfFinancing(unittest.TestCase):

    def setUp(self):
        years = ["2015", "2014", "2013", "2012", "2011"]
        self.income = Mock()
        self.balance = Mock()
        self.cash = Mock()
        
        self.cash.dividends_total = pandas.Series([12784.7, 9000.0, 9000.0, 12000.0, 19000.0], index = years)
        self.income.num_shares_diluted = pandas.Series([153646.0, 152881.0, 151569.0, 150000.0, 125932.0], index = years)

        self.adjusted_earnings = pandas.Series([11150.46, 7893.12, 5666.38, 12404.10, 19816.02], index = years)
        self.invested_capital = pandas.Series([130000.0, 120000, 67000.0, 124000, 195000.0], index = years)

        self.financials = FinanceAnalyst(self.income, self.balance, self.cash)
        self.financials.ownerEarnings = Mock(return_value = self.adjusted_earnings)
        self.financials.investedCapital = Mock(return_value = self.invested_capital)

    
    def test_DividendRate(self):
        expected = self.adjusted_earnings / self.cash.dividends_total
        div_rate = self.financials.dividend_rate()
        self.assertTrue(div_rate.equals(expected))

    def test_CapitalInvestmentPercentage(self):
        expected = (self.adjusted_earnings - self.cash.dividends_total) / self.invested_capital
        invest_pct = self.financials.capitalInvestmentPct()
        self.assertTrue(invest_pct.equals(expected))

    def test_DilutionGrowth(self):
        shares_growth = self.income.num_shares_diluted.pct_change(periods = -1)
        dilution_growth = self.financials.dilutionGrowth()
        self.assertEqual(dilution_growth, shares_growth.mean())
        self.assertAlmostEqual(dilution_growth, 0.0538, places = 4)


class Test_CalculateWACC(unittest.TestCase):
    
    def setUp(self):
        years = ["2015", "2014", "2013", "2012", "2011"]
        self.income = Mock()
        self.balance = Mock()
        self.cash = Mock()
        
        self.financial_analyst = FinanceAnalyst(self.income, self.balance, self.cash)
        self.adjusted_earnings = pandas.Series([13863.84, 11931.48, 6634.01, 12389.30, 19480.11], index = years)
        self.invested_capital = pandas.Series([130000.0, 120000, 67000.0, 124000, 195000.0], index = years)
        self.assets = pandas.Series([102972.0, 81903.0, 63604.2, 64117.2, 68824.9], index = years)
        self.debt = pandas.Series([1104.3, 535.4, 908.3, 479.0, 928.5], index = years)
        self.free_cash = pandas.Series([140.3, 52.2, 85.3, 44.0, 28.5], index = years)
        self.financial_analyst.ownerEarnings = Mock(return_value = self.adjusted_earnings)
        self.financial_analyst.investedCapital = Mock(return_value = self.invested_capital)
        self.financial_analyst.totalAssets = Mock(return_value = self.assets)
        self.financial_analyst.totalDebt = Mock(return_value = self.debt)
        self.financial_analyst.netCash = Mock(return_value = self.free_cash)

        self.analyser = EPVanalyser(self.financial_analyst)

    def test_AdjustedReturnOnInvestedCapital(self):
        adj_ROIC = self.analyser.owner_earnings() / self.financial_analyst.investedCapital()
        trend_ROIC = self.financial_analyst.series_trend(adj_ROIC)
        mean_rtn = trend_ROIC["2015"]
        expected_std = (adj_ROIC - trend_ROIC).std()
        expected_mean = mean_rtn - 1.65 * (expected_std / (5.0 ** 0.5))
        adj_ROIC_mean = self.analyser.ROIC_mean()
        adj_ROIC_std = self.analyser.ROIC_std()
        self.assertEqual(adj_ROIC_std, expected_std)
        self.assertEqual(adj_ROIC_mean, expected_mean)

    def test_OptimalF(self):
        min_denom = 0.01
        adj_frac = 1.0 / 6.0
        mean = self.analyser.ROIC_mean()
        std = self.analyser.ROIC_std()
        expectedF = adj_frac * mean / max(std ** 2 - mean ** 2, min_denom)
        self.assertEqual(self.analyser.optF(), expectedF)

    def test_OptimalFwithNegReturn(self):
        min_denom = 0.01
        adj_frac = 1.0 / 6.0
        min_F = 0.001
        self.analyser.ROIC_mean = Mock(return_value = -0.25)
        mean = self.analyser.ROIC_mean()
        std = self.analyser.ROIC_std()
        expectedF = max(adj_frac * mean / max(std ** 2 - mean ** 2, min_denom), min_F)
        self.assertEqual(self.analyser.optF(), expectedF)

    def test_ActualF(self):
        current_debt = self.debt["2015"]
        current_free_cash = self.free_cash["2015"]
        net_debt = current_debt - current_free_cash
        current_assets = self.assets["2015"]
        expectedF = current_assets / (current_assets - net_debt)
        self.assertEqual(self.analyser.actF(), expectedF)

    def test_WACCbaseline(self):
        debt_cost = 0.09
        equity_cost = 0.20
        self.analyser.debt_cost = debt_cost
        self.analyser.equity_premium = 0.11
        self.analyser.equity_base_cost = 0.09
        f_opt = 3.5
        self.analyser.optF = Mock(return_value = f_opt)
        self.assertEqual(self.analyser.WACC(equity_cost), equity_cost * (1 / f_opt) + debt_cost * ((f_opt - 1) / f_opt))

    def test_ProbabilityOfLoss(self):
        acceptable_DD = 0.25
        self.analyser.acceptable_DD = acceptable_DD
        mean_rtn = 0.212
        std_rtn = 0.159
        self.analyser.ROIC_mean = Mock(return_value = mean_rtn)
        self.analyser.ROIC_std = Mock(return_value = std_rtn)
        optF = 3.5
        expected_prob = math.exp((-2.0 * optF * mean_rtn * acceptable_DD) / (optF * std_rtn) ** 2)
        self.assertEqual(self.analyser.loss_probability(optF), expected_prob)

    def test_LeveragedEquityCost(self):
        prob_at_opt = 0.301
        prob_at_act = 0.016
        loss_prob_mock = Mock()
        loss_prob_mock.side_effect = [prob_at_opt, prob_at_act]
        self.analyser.loss_probability = loss_prob_mock
        leverage_ratio = prob_at_act / prob_at_opt
        equity_base_cost = 0.09
        equity_premium = 0.11
        self.analyser.equity_premium = equity_premium
        self.analyser.equity_base_cost = equity_base_cost
        expected_cost = equity_base_cost + leverage_ratio * equity_premium
        equity_cost = self.analyser.equity_cost()
        self.assertEqual(equity_cost, expected_cost)
        self.assertAlmostEqual(equity_cost, 0.096, places = 3)
        

class Test_GrowthMultiple(unittest.TestCase):

    def setUp(self):
        years = ["2015", "2014", "2013", "2012", "2011"]
        self.income = Mock()
        self.balance = Mock()
        self.cash = Mock()

        self.financial_analyst = FinanceAnalyst(self.income, self.balance, self.cash)
        self.analyser = EPVanalyser(self.financial_analyst)
        self.adjusted_earnings = pandas.Series([11150.46, 7893.12, 5666.38, 12404.10, 19816.02], index = years)
        self.invested_capital = pandas.Series([41642.4, 50073.4, 49105.9, 32548.6, 34948.5], index = years)
        self.financial_analyst.ownerEarnings = Mock(return_value = self.adjusted_earnings)
        self.financial_analyst.investedCapital = Mock(return_value = self.invested_capital)


    def test_GrowthMultiple(self):
        WACC = 0.093
        R = 0.212 # mean return
        I = 0.026 # cash investment percentage
        expected = (1 - (I / WACC)*(WACC / R)) / (1 - (I / WACC))
        growth_mult = self.analyser.growth_multiple(WACC, I, R)
        self.assertEqual(growth_mult, expected)
        self.assertAlmostEqual(growth_mult, 1.218, places = 3)

    def test_GrowthMultipleCap(self):
        WACC = 0.10
        R = 0.15
        I = 0.14
        expected = (1 - 0.75 * WACC / R) / (1 - 0.75)
        growth_mult = self.analyser.growth_multiple(WACC, I, R)
        self.assertAlmostEqual(growth_mult, 2.000, places = 3)

    def test_GrowthMultipleNegRtn(self):
        WACC = 0.25
        R = -0.025
        I = 0.05
        expected = (1 - (I / WACC) * abs(WACC / R)) / (1 - (I / WACC))
        growth_mult = self.analyser.growth_multiple(WACC, I, R)
        self.assertAlmostEqual(growth_mult, -1.250, places = 3)


class Test_AdjustmentForCyclicality(unittest.TestCase):

    def setUp(self):
        years = ["2015", "2014", "2013", "2012", "2011"]
        self.income = Mock()
        self.balance = Mock()
        self.cash = Mock()

        self.income.sales = pandas.Series([216892.6, 114182.9, 114695.4, 152837.9, 142511.6], index = years)
        self.income.COGS = pandas.Series([193088.80, 94391.70, 96937.40, 128595.40, 106295.80], index = years)
        self.income.SGA = pandas.Series([8135.10, 7170.40, 7966.20, 6447.00, 8320.20], index = years)
        self.income.DandA = pandas.Series([4169.4, 1639.2, 974.8, 685.7, 542.4], index = years)

        self.financial_analyst = FinanceAnalyst(self.income, self.balance, self.cash)
        self.analyser = EPVanalyser(self.financial_analyst)
        self.adjusted_earnings = pandas.Series([11150.46, 7893.12, 5666.38, 12404.10, 19816.02], index = years)
        self.maintenance_capex = pandas.Series([2287.83, 1973.08, 1658.33, 1343.58, 1028.83], index = years)
        self.invested_capital = pandas.Series([41642.39, 50073.44, 49105.93, 32548.57, 34948.47], index = years)
        self.adjusted_ROIC_mean = 0.196
        self.financial_analyst.ownerEarnings = Mock(return_value = self.adjusted_earnings)
        self.financial_analyst.expended_depreciation = Mock(return_value = self.maintenance_capex)
        self.financial_analyst.investedCapital = Mock(return_value = self.invested_capital)
        self.analyser.ROIC_mean = Mock(return_value = self.adjusted_ROIC_mean)


    def test_TrendEarnings(self):
        expected = self.financial_analyst.series_trend(self.adjusted_earnings)
        trend_earnings = self.financial_analyst.trendEarnings()
        self.assertTrue(trend_earnings.equals(expected))

    def test_EarningsOnROIC(self):
        expected = self.invested_capital * self.adjusted_ROIC_mean
        adj_earnings = self.analyser.ROIC_adjusted_earnings()
        self.assertTrue(adj_earnings.equals(expected))


class Test_EPVcalcs(unittest.TestCase):

    def setUp(self):
        years = ["2015", "2014", "2013", "2012", "2011"]
        self.income = Mock()
        self.balance = Mock()
        self.cash = Mock()

        self.financial_analyst = FinanceAnalyst(self.income, self.balance, self.cash)
        self.analyser = EPVanalyser(self.financial_analyst)
        self.adjusted_earnings = pandas.Series([8143.91, 9792.75, 6378.77, 6365.45, 6834.80], index = years)
        self.trend_earnings = pandas.Series([75010.89, 61676.55, 48342.21, 35007.87, 21673.53], index = years)
        self.net_cash = pandas.Series([61329.61, 31829.56, 14498.27, 31568.63, 33876.43], index = years)
        self.debt = pandas.Series([1104.3, 535.4, 908.3, 479.0, 928.5], index = years)
        self.num_shares_diluted = pandas.Series([153646.0, 152881.0, 151569.0, 150000.0, 125932.0], index = years)
        self.financial_analyst.ownerEarnings = Mock(return_value = self.adjusted_earnings)
        self.financial_analyst.trendEarnings = Mock(return_value = self.trend_earnings)
        self.financial_analyst.netCash = Mock(return_value = self.net_cash)
        self.financial_analyst.totalDebt = Mock(return_value = self.debt)
        self.financial_analyst.numSharesDiluted = Mock(return_value = self.num_shares_diluted)


    def test_EPVcalculation(self):
        earnings = self.adjusted_earnings
        WACC = 0.0945
        growth = 1.057
        dilution = 0.9462
        expected = ((earnings / WACC) * growth - self.debt + self.net_cash) * dilution
        EPV = self.analyser.EPV(earnings, WACC, growth, dilution)
        self.assertTrue(EPV.equals(expected))
        self.assertAlmostEqual(EPV["2015"], 143175.63, places = 2)

    def test_EPVcalculationBase(self):
        earnings = self.adjusted_earnings
        WACC = 0.0945
        expected = (earnings / WACC) - self.debt + self.net_cash
        EPV = self.analyser.EPV(earnings, WACC, 1, 1)
        self.assertTrue(EPV.equals(expected))
        self.assertAlmostEqual(EPV["2015"], 146404.25, places = 2)

    def test_EPVcalculationTable(self):
        earnings = pandas.DataFrame({"Base" : self.adjusted_earnings, 
                                     "Other" : self.adjusted_earnings})
        WACC = 0.0945
        expected = (earnings / WACC).sub(self.debt, axis = "index").add(self.net_cash, axis = "index")
        EPV = self.analyser.EPV(earnings, WACC, 1, 1)
        self.assertTrue(EPV.equals(expected))
        self.assertAlmostEqual(EPV["Base"]["2015"], 146404.25, places = 2)

    def test_EPVperShare(self):
        earnings = self.adjusted_earnings
        WACC = 0.0945
        EPV = self.analyser.EPV(earnings, WACC, 1, 1)
        expected = EPV / self.num_shares_diluted
        EPV_share = self.analyser.per_share(EPV)
        self.assertTrue(EPV_share.equals(expected))
        self.assertAlmostEqual(EPV_share["2015"], 0.953, places = 3)

    def test_EPVperShareTable(self):
        earnings = pandas.DataFrame({"Base" : self.adjusted_earnings, 
                                     "Other" : self.adjusted_earnings})
        WACC = 0.0945
        EPV = self.analyser.EPV(earnings, WACC, 1, 1)
        expected = EPV.div(self.num_shares_diluted, axis = "index")
        EPV_share = self.analyser.per_share(EPV)
        self.assertTrue(EPV_share.equals(expected))
        self.assertAlmostEqual(EPV_share["Base"]["2015"], 0.953, places = 3)


class Test_EPVvariations(unittest.TestCase):

    def setUp(self):
        self.earnings = "adjusted_earnings"
        self.trend = "trend_earnings"
        self.ROIC_earnings = "ROIC_earnings"
        self.min_earnings = "min_earnings"
        self.max_earnings = "max_earnings"
        self.earnings_table = Mock()
        self.earnings_table.min = Mock(return_value = self.min_earnings)
        self.earnings_table.max = Mock(return_value = self.max_earnings)
        return_series = pandas.Series([0] * 3)
        self.growth = 2.0
        self.dilution = 1 - 0.05
        self.WACC = 0.12
        self.WACC_base = 0.18
        self.analyser = EPVanalyser(Mock())

        self.analyser.owner_earnings = Mock(return_value = self.earnings)
        self.analyser.trend_earnings = Mock(return_value = self.trend)
        self.analyser.ROIC_adjusted_earnings = Mock(return_value = self.ROIC_earnings)
        self.analyser.earnings_table = Mock(return_value = self.earnings_table)
        self.analyser.EPV = Mock(return_value = return_series)
        self.analyser.WACC = Mock(return_value = self.WACC)
        self.analyser.WACC_base = Mock(return_value = self.WACC_base)
        self.analyser.dilution = Mock(return_value = self.dilution)
        self.analyser.growth_multiple = Mock(return_value = self.growth)


    def test_EPVbase(self):
        EPV = self.analyser.EPV_base()
        self.analyser.EPV.assert_called_once_with(self.earnings, self.WACC_base, growth = 1, dilution = 1)

    def test_EPVlevered(self):
        EPV = self.analyser.EPV_levered()
        self.analyser.EPV.assert_called_once_with(self.earnings, self.WACC, growth = 1, dilution = 1)

    def test_EPVgrowth(self):
        EPV = self.analyser.EPV_growth()
        self.analyser.EPV.assert_called_once_with(self.earnings, self.WACC_base, growth = self.growth, dilution = 1)

    def test_EPVcyclic(self):
        EPV = self.analyser.EPV_cyclic()
        self.analyser.EPV.assert_called_once_with(self.ROIC_earnings, self.WACC_base, growth = 1, dilution = 1)

    def test_EPVdiluted(self):
        EPV = self.analyser.EPV_diluted()
        self.analyser.EPV.assert_called_once_with(self.earnings, self.WACC_base, growth = 1, dilution = self.dilution)

    def test_EPVadjusted(self):
        EPV = self.analyser.EPV_adjusted()
        self.analyser.EPV.assert_called_once_with(self.earnings, self.WACC, growth = self.growth, dilution = self.dilution)

    def test_EPVmin(self):
        EPV = self.analyser.EPV_minimum()
        self.analyser.EPV.assert_called_once_with(self.min_earnings, self.WACC_base, 1, self.dilution)

    def test_EPVmax(self):
        EPV = self.analyser.EPV_maximum()
        self.analyser.EPV.assert_called_once_with(self.max_earnings, self.WACC, self.growth, 1)



if __name__ == '__main__':
    unittest.main()
