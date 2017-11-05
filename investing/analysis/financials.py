import pandas
import math
import datetime
from dateutil.relativedelta import relativedelta
import warnings
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import sys
import os
sys.path.append(os.path.join("C:\\Users", os.getlogin(), "Source\\Repos\\FinancialDataHandling\\financial_data_handling"))

from formats.fundamentals import Financials, ValuationSummary, Valuations
from formats.price_history import PriceHistory
from store.file_system import Storage
from multiprocessing import Process, Queue, Pool


# REFINEMENTS
# TODO - Note VSC has updated financials (for Dec 2015), will be a good test case to see if updating works.
# TODO - Growth multiple doesn't seem to work well. Sometimes is higher than appears justified.
# TODO - Add stock blacklist to avoid cycling through those with errors every time.

# REFACTORING
# TODO - Move series manipulations (trend, diff etc) to object derived from pandas.Series
# TODO - Statement can create this object on get_row

# FIXES
# TODO - Improve error handling in storeValuationSummaryBrief
# TODO - Apply appropriate corrections when not all years have reported values.
# TODO - Apply to: Capital Base, Total Assets
#
# Errors:
# TODO - Account for USD reported accounts e.g. RMD
#
# Aliases:
# TODO - "Cash & Short Term Investments" - "Cash Only" (AMP, IAG, ISU, MPL, QBE, SUN)
# TODO - Handle statements for financial companies with no "Sales/Revenue":
# TODO - ANZ, ABA, BOQ, BEN, CBA, GMY, IQE, KMD, KNH, MQG, WBC
#
# TODO - Data not available: KMD, KNH, MEZ, MPP, NEC, NPX, SNC, SFL, TGZ

# FEATURES
# TODO - Update financials for new reporting period.
# TODO - Add Book Valuations.
# DONE - Parallel processing of valuations.
# DONE - Update valuations based on latest prices only.
# TODO - Analysis of financial stocks - Banks and Insurance.
# TODO - Add data from analyst forecast earnings.
# TODO - Add data from competitors and industry.


def retrieveOverviewData(storage_dir, headings = None):
    store = Storage(storage_dir)
    xls = XLSio(store)
    scraper = WSJscraper()
    xls.loadWorkbook("ASXListedCompanies")
    tickers = xls.getTickers()
    new_data = {}
    for ticker in tickers:
        scraper.load_overview(ticker, store)
        try:
            new_data[ticker] = scraper.keyStockData()
        except Exception:
            print("Problem with: " + ticker)
    xls.updateTable(new_data)
    xls.saveAs("StockSummary")

# TODO - Move price updates to Collator
def updatePrices(date = None):
    store = Storage()
    if date is None:
        files = store.list_files(store.valuationSummary(""), "Valuation")
        dates = [val[-13:-5] for val in files]
        date = max(dates)
    valuations = store.load(ValuationSummary(date))
    summary = valuations.summary
    download = WebDownloader()
    new_prices = pandas.Series([download.currentPrice(ticker) for ticker in summary.index], index = summary.index)
    old_prices = summary["Current Price"]
    values = summary.loc[:, "Adjusted Value":"Dilution Value"]
    new_values = (values + 1).multiply(old_prices / new_prices, axis = "rows") - 1
    summary.loc[:, "Adjusted Value":"Dilution Value"] = new_values
    summary["Current Price"] = new_prices
    summary["Current PE"] = summary["Current EPS"] / summary["Current Price"]
    valuations.summary = summary
    store.save(valuations)
    return summary


def collatePriceChanges(all_valuations, type, periods = [6, 12, 24]):

    tickers = all_valuations["ticker"].unique()
    downloader = WebDownloader()

    results = pandas.DataFrame()

    for ticker in tickers:
        analyser = PriceAnalyser(downloader.priceHistory(ticker))
        valuations = all_valuations[all_valuations["ticker"] == ticker]
        results = results.append(analyser.price_appreciation(ticker, valuations, type, periods))
        
    return results


class Collator():

    def __init__(self, exchange, tickers):
        self.tickers = tickers
        self.store = Storage(exchange)
        self.reporter = Reporter(None, Factory(exchange))

    def collateValuations(self, num_processes = 8):
        if self.tickers is None:
            xls = XLSio(self.store)
            xls.loadWorkbook("StockSummary.xlsx")
            xls.table = xls.table[xls.table["P/E Ratio (TTM)"].notnull()]
            self.tickers = xls.getTickers()
    
        pool = Pool(processes = num_processes)
        all_results = pool.map(self.get_valuations, self.tickers)
        errors = []
        results = pandas.DataFrame()

        for result in all_results:
            if type(result) is pandas.DataFrame:
                results = results.append(result)
            else:
                errors.append(result)

    
        print(str(len(errors)) + " Failed")
        print(str(len(results["ticker"].unique())) + " Succeeded")

        valuations = Valuations(datetime.datetime.strftime(datetime.date.today(), "%Y%m%d"))
        valuations.summary = results
        self.store.save(valuations)

        return errors

    def get_valuations(self, ticker):
        try:
            self.reporter.analyse(ticker)
            result = self.reporter.share_values
            result.insert(0, "ticker", ticker)
        except Exception as E:
            result = "Error valuing {0}: {1}".format(ticker, E)
        return result


    def parallelSummary(self, num_processes):
        if self.tickers is None:
            xls = XLSio(self.store)
            xls.loadWorkbook("StockSummary.xlsx")
            xls.table = xls.table[xls.table["P/E Ratio (TTM)"].notnull()]
            self.tickers = xls.getTickers()
    
        pool = Pool(processes = num_processes)
        all_results = pool.map(self.get_one_line_summary, self.tickers)
        errors = []
        results = []

        for result in all_results:
            if type(result) is pandas.Series:
                results.append(result)
            else:
                errors.append(result)

        valuation_summary = ValuationSummary(datetime.datetime.strftime(datetime.date.today(), "%Y%m%d"))
        valuation_summary.summary = pandas.concat(results, axis = 1).T
        self.store.save(valuation_summary)
    
        print(str(len(errors)) + " Failed")
        print(str(len(results)) + " Succeeded")

        return errors


    def get_one_line_summary(self, ticker):
        try:
            self.reporter.analyse(ticker)
            result = self.reporter.oneLineValuation()
        except Exception as E:
            result = "Error valuing {0}: {1}".format(ticker, E)
        return result

        

class Factory():

    def __init__(self, exchange):
        self.exchange = exchange
        self.store = Storage(exchange)

    def buildFinancialAnalyst(self, ticker):
        annual = self.store.load(Financials(ticker, "annual"))
        interim = self.store.load(Financials(ticker, "interim"))
        income = IncomeStatement(annual, interim)
        balance = BalanceSheet(annual, interim)
        cashflow = CashflowStatement(annual, interim)
        return FinanceAnalyst(income, balance, cashflow)

    
    def buildPriceAnalyser(self, ticker):
        price_history = PriceHistory(ticker)
        self.store.load(price_history)
        return PriceAnalyser(price_history.prices) 


# TODO Reporter analysis including current price needs updating - currently only uses last saved price, not latest price.
class Reporter():
    
    def __init__(self, ticker = None, factory = Factory("ASX")):
        self.factory = factory

        if ticker is not None:
            self.analyse(ticker)

    def analyse(self, ticker):
        '''
        Analyse preps the reporter to assess the current value of the stock relative to 
        the current price - e.g. to provide the full summary table, or one line summary.
        '''
        self._index = None
        self._maintenance = None
        self._earnings = None
        self._EPV = None
        self._prices = None
        self._values = None
        self.ticker = ticker
        self.financials = self.factory.buildFinancialAnalyst(ticker)
        self.EPVanalyser = EPVanalyser(self.financials)
        self.price_analyser = self.factory.buildPriceAnalyser(ticker)
        self.current_price = self.price_analyser.latest_price

    def evaluate(self, ticker):
        '''
        Evaluate preps the report to just conduct the valuations with the currently available
        financial data. This will allow EPV reporting but not an assessment of the current value
        relative to the price. This is used for Valuations creation.
        '''
        self._index = None
        self._maintenance = None
        self._earnings = None
        self._EPV = None
        self._prices = None
        self._values = None
        self.ticker = ticker
        self.financials = self.factory.buildFinancialAnalyst(ticker)
        self.EPVanalyser = EPVanalyser(self.financials)

    def financialsToExcel(self, writer):
        self.financials.to_excel(writer)

    @property
    def index(self):
        if self._index is None:
            self._index = self.earnings.index
        return self._index
        
    @property
    def EPV(self):
        if self._EPV is None:
            self._EPV = self.EPV_table()
        return self._EPV

    @property
    def earnings(self):
        if self._earnings is None:
            self._earnings = self.earnings_table()
        return self._earnings

    @property
    def maintenance(self):
        if self._maintenance is None:
            self._maintenance = self.maintenance_table()
        return self._maintenance

    @property
    def prices(self):
        if self._prices is None:
            self._prices = self.prices_table()
        return self._prices

    @property
    def share_values(self):
        if self._values is None:
            self._values = self.shareValue_table()
        return self._values

    @property
    def EPS(self):
        EPS = self.financials.diluted_EPS()
        return EPS.round(3)

    @property
    def DPS(self):
        DPS = self.financials.DPS_common()
        return DPS.round(3)

    @property
    def PE(self):
        PE = self.prices.div(self.EPS, axis = "index")
        return PE.round(1)
 
    def EPV_table(self):
        table = pandas.DataFrame()
        table["Adjusted"] = self.EPVanalyser.EPV_adjusted()
        table["Min"] = self.EPVanalyser.EPV_minimum()
        table["Max"] = self.EPVanalyser.EPV_maximum()
        table["Base"] = self.EPVanalyser.EPV_base()
        table["Levered"] = self.EPVanalyser.EPV_levered()
        table["Growth"] = self.EPVanalyser.EPV_growth()
        table["Cyclic"] = self.EPVanalyser.EPV_cyclic()
        table["Dilution"] = self.EPVanalyser.EPV_diluted()
        return table

    def maintenance_table(self):
        table = pandas.DataFrame()
        table["PPE maintenance"] = self.financials.PPE_maintenance()
        table["Intangibles maintenance"] = self.financials.intangibles_maintenance()
        table["Working capital"] = self.financials.working_capital_requirements()
        table["Total maintenance"] = self.financials.maintenance_expense()
        table["Capital injections"] = self.financials.capital_injections()
        table["Expended Dep."] = self.financials.expended_depreciation()
        table["Dep and Amor"] = self.financials.non_cash_expenses()
        table["Applied cash expense"] = self.financials.cash_expenses()
        return table.round(1)

    def earnings_table(self):
        table = self.EPVanalyser.earnings_table()
        return table.round(1)

    def valuationMetrics_table(self):
        table = pandas.DataFrame()
        table["ROIC (%)"] = self.format_pct(self.EPVanalyser.ROIC())
        table["Mean Return (%)"] = self.format_pct(self.EPVanalyser.ROIC_mean())
        table["Std Dev Return (%)"] = self.format_pct(self.EPVanalyser.ROIC_std())
        table["opt F"] = self.format_ratio(self.EPVanalyser.optF())
        table["act F"] = self.format_ratio(self.EPVanalyser.actF())
        table["WACC base (%)"] = self.format_pct(self.EPVanalyser.WACC_base())
        table["WACC adj. (%)"] = self.format_pct(self.EPVanalyser.WACC())
        table["Growth mult."] = self.format_ratio(self.EPVanalyser.growth_multiple())
        table["Dilution (%)"] = self.format_pct(self.EPVanalyser.dilution())
        return table

    def shareValue_table(self):
        values = self.EPVanalyser.per_share(self.EPV)
        return values.round(3)

    def prices_table(self):
        prices = self.price_analyser.price_table(self.index)
        return prices.round(3)

    def valuation_pct(self):
        valuation = self.share_values.iloc[-1]
        if self.current_price != 'N/A':
            valuation = (valuation / self.current_price - 1)
            valuation["All +ve"] = all(valuation > 0)
        return valuation

    def summaryTable(self):
        costs = self.make_section(self.maintenance, "Maintenance, (thou's)")
        earnings = self.make_section(self.earnings, "Earnings (thou's)")
        metrics = self.make_section(self.valuationMetrics_table(), "Valuation Metrics")
        EPV = self.make_section((self.EPV / 1000.0).round(1), "EPV (mill's)")
        value = self.make_section(self.share_values, "Share Valuation")
        prices = self.make_section(self.prices, "Price history")
        EPS = self.make_section(self.EPS, "EPS")
        PE = self.make_section(self.PE, "PE history")
        current = self.current_price_section()
        table = pandas.concat([costs, earnings, metrics, EPV, value, prices, EPS, PE, current], 
                              axis = 1)
        table = table.iloc[::-1]
        return table.T

    def oneLineValuation(self):
        summary = pandas.Series(dtype = float)
        if self.current_price == 'N/A':
            summary["Current Price"] = np.nan
        else:
            summary["Current Price"] = self.current_price
        summary["Current EPS"] = self.EPS[-1]
        summary["Current Div"] = self.DPS[-1]
        summary["Average Div"] = self.DPS.mean()
        summary = summary.append(self.valuationMetrics_table().iloc[-1])
        valuation = self.valuation_pct().round(3)
        valuation = self.append_label(valuation, "Value")
        summary = summary.append(valuation)
        summary = summary.append(self.PE_ranges().round(1))
        summary.name = self.ticker
        return summary

    def PE_ranges(self):
        PE_summary = pandas.Series(dtype = float)
        PE_summary["Current PE"] = self.current_price / self.EPS[-1]
        PE_current = self.PE.iloc[-1]
        PE_current = self.append_label(PE_current, "PE Current Yr")
        PE_summary = PE_summary.append(PE_current)
        PE_summary["Average PE 5yr"] = self.PE.Average.mean()
        PE_summary["High PE 5yr"] = self.PE.High.max()
        PE_summary["Low PE 5yr"] = self.PE.Low.min()
        PE_summary["Std Dev PE 5yr"] = self.PE["Std Dev"].mean()
        return PE_summary

    def LeibowitzPEsummary(self):
        table = pandas.DataFrame()
        mean_rtn = self.EPVanalyser.ROIC_mean()
        capital_cost = self.EPVanalyser.WACC()
        growth_multiple = self.EPVanalyser.growth_multiple()
        franchise_factor = (mean_rtn - capital_cost) / (capital_cost * mean_rtn)
        theoretical_PE = (1 / capital_cost) + franchise_factor * growth_multiple
        book_value = self.EPVanalyser.book_value()
        earnings_value = self.EPVanalyser.per_share(self.EPV["Base"])
        franchise_value = (1 / capital_cost) * (mean_rtn - capital_cost) * (growth_multiple * book_value)

        table["Mean Return (%)"] = self.format_pct(mean_rtn)
        table["Franchise Return (%)"] = self.format_pct(mean_rtn)
        table["WACC base (%)"] = self.format_pct(self.EPVanalyser.WACC_base())
        table["WACC adj. (%)"] = self.format_pct(capital_cost)
        table["Growth mult."] = self.format_ratio(growth_multiple)
        table["Franchise Factor (FF)"] = self.format_ratio(franchise_factor)
        table["Theoretical PE"] = self.format_ratio(theoretical_PE)
        table["Book Value ($)"] = self.format_ratio(book_value)
        table["Earnings Value ($)"] = self.format_ratio(earnings_value)
        table["Franchise Value ($)"] = self.format_ratio(franchise_value)

        return table

    def append_label(self, series, label):
        series.index = [" ".join([ix, label]) for ix in series.index]
        return series

    def make_section(self, table, name):
        heading = self.heading(name)
        section = pandas.concat([heading, table], axis = 1)
        return section

    def current_price_section(self):
        row = self.pad_row(self.current_price)
        date_today = str(datetime.date.today())
        price_table = pandas.DataFrame({date_today : row})
        return self.make_section(price_table, "Current Price")

    def heading(self, name):
        header = pandas.Series(["*****"] * len(self.index), index = self.index)
        header.name = name
        return header
   
    def pad_row(self, value):
        return pandas.Series((["-"] * (len(self.index) - 1) + [value]), index = self.index)

    def format_pct(self, series):
        return series.apply(lambda x: round(100.0 * x, 2))

    def format_ratio(self, series):
        return series.apply(lambda x: round(x, 2))

    def valueHistogram(self):
        valuation = self.valuation_pct()
        valuation.pop("All +ve")
        ax = valuation.plot(kind = "bar")
        ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in ax.get_yticks()])
        plt.xticks(rotation = 45)
        plt.subplots_adjust(bottom = 0.15)
        plt.show()

    def valueVSprice(self):
        values = self.share_values
        xlabels = [timestamp.strftime("%Y-%b") for timestamp in values.index]
        prices = self.prices
        value_points = plt.plot(values, 'o')
        plt.legend(value_points, values.columns.tolist(), 
                   loc = 'upper center', 
                   fancybox = True, 
                   bbox_to_anchor = (0.5, 1.05), 
                   ncol = 4, 
                   prop = {'size' : 10})
        plt.plot(prices["Average"], 'r')
        plt.plot(prices["High"], "r--")
        plt.plot(prices["Low"], "r--")
        plt.plot(len(xlabels) - 1, self.current_price, "rs")
        ax = plt.axes()
        ax.set_yticklabels(['$%s' % float('%.3g' % x) for x in ax.get_yticks()])
        plt.xlim(-0.5, len(xlabels) - 0.5)
        plt.xticks(range(len(xlabels)), xlabels)
        plt.xticks(rotation = 45)
        plt.subplots_adjust(bottom = 0.15)
        plt.show()



class EPVanalyser():

    def __init__(self, finance_analyst):
        self.financials = finance_analyst
        self.debt_cost = 0.09
        self.equity_base_cost = 0.09
        self.equity_premium = 0.11
        self.acceptable_DD = 0.25
    

    def owner_earnings(self):
        return self.financials.ownerEarnings()

    def trend_earnings(self):
        return self.financials.trendEarnings()

    def ROIC_adjusted_earnings(self):
        return self.financials.investedCapital() * self.ROIC_mean()

    def dilution(self):
        return 1 - self.financials.dilutionGrowth()

    def book_value(self):
        return self.per_share(self.financials.shareholdersEquity())

    def ROIC(self):
        return self.financials.ROIC()

    def ROIC_mean(self):
        return self.financials.ROIC_mean()

    def ROIC_std(self):
        return self.financials.ROIC_std()

    def optF_theoretical(self):
        min_denom = 0.001
        adj_frac = 1.0 / 6.0
        min_F = 0.001
        mean = self.ROIC_mean()
        std = self.ROIC_std()
        denom = std ** 2 - mean ** 2
        denom[denom < min_denom] = min_denom
        optF = adj_frac * mean / denom
        optF[optF < min_F] = min_F
        return optF

    def optF_simplified(self):
        mean = self.ROIC_mean()
        std = self.ROIC_std()
        adj_frac = 1.0 / 6.0
        min_F = 0.001
        optF = adj_frac * mean / (std ** 2)
        optF[optF < min_F] = min_F
        return optF

    def optF_max_allowable_loss(self):
        mean = self.ROIC_mean()
        std = self.ROIC_std()
        three_sigma_loss = mean - 3 * std
        optF = -1 / three_sigma_loss
        optF[three_sigma_loss >= 0] = 10
        return optF

    def optF(self):
        optF = pandas.DataFrame([self.optF_theoretical(), self.optF_simplified(), self.optF_max_allowable_loss()]).min()
        return optF

    def actF(self):
        # Actual leverage is calculated as if all free cash is used to pay down debt.
        assets = self.financials.totalAssets()
        debt = self.financials.totalDebt()
        net_cash = self.financials.netCash()
        return (assets - net_cash) / (assets - debt)

    def WACC(self, equity_cost = None):
        if equity_cost is None:
            equity_cost = self.equity_cost()
        optF = self.optF()
        return equity_cost * (1.0 / optF) + self.debt_cost * ((optF - 1.0) / optF)

    def WACC_base(self):
        equity_cost = self.equity_base_cost + self.equity_premium
        return self.WACC(equity_cost)

    def equity_cost(self):
        prob_opt = self.loss_probability(self.optF())
        prob_act = self.loss_probability(self.actF())
        leverage_ratio = prob_act / prob_opt
        leverage_ratio[prob_opt > 0.05] = 1
        return self.equity_base_cost + leverage_ratio * self.equity_premium

    def loss_probability(self, optF):
        mean_rtn = self.ROIC_mean()
        std_rtn = self.ROIC_std()
        exponent = (-2.0 * optF * mean_rtn * self.acceptable_DD) / (optF * std_rtn) ** 2
        exponent[exponent > 10] = 10
        probability = exponent.apply(math.exp)
        return probability

    def growth_multiple(self, WACC = None, invest_pct = None, mean_rtn = None):
        if WACC is None:
            WACC = self.WACC()
        if invest_pct is None:
            invest_pct = self.financials.capitalInvestmentPct()
        if mean_rtn is None:
            mean_rtn = self.ROIC_mean()
        # Ref: "Value Investing" Greenwald, Kahn et al pg 143
        # Note numerous adjustments are conducted to make sure the multiple is greater than 0.
        # These are required as the calculation doesn't work well when invest_pct or mean_rtn
        # are negative, or when mean_rtn is less than WACC.
        # The calculation attempts to give multiples between 0 and 1, when growth is bad.
        # Note that negative growth multiple is not desirable, as this would give positive EPV
        # when multiplied with negative earnings.
        invest_WACC_ratio = invest_pct / WACC
        invest_WACC_ratio[invest_WACC_ratio > 0.75] = 0.75
        invest_WACC_ratio[(WACC > mean_rtn) & (invest_WACC_ratio < 0)] = 0.75
        # Ratio of WACC to Return adjusted for neg return conditions.
        WACC_rtn_ratio = WACC / mean_rtn
        zero_returns = (mean_rtn == 0)
        if any(zero_returns):
            WACC_rtn_ratio[zero_returns] = 2
        neg_returns = (mean_rtn < 0)
        if any(neg_returns):
            WACC_rtn_ratio[neg_returns] = ((WACC[neg_returns] - mean_rtn[neg_returns]) / mean_rtn[neg_returns]).abs()
            WACC_rtn_ratio[neg_returns & (WACC_rtn_ratio > 1.5)] = 1.5
        
        multiple = (1 - invest_WACC_ratio * WACC_rtn_ratio) / (1 - invest_WACC_ratio)
        # If cost of capital (WACC) is greater than mean_rtn, then growth is bad.
        # We want to return a multiple less than 1.
        bad_ratio = (WACC_rtn_ratio > 1)
        if any(bad_ratio):
            bad_WACC_rtn_ratio = 1 / WACC_rtn_ratio[bad_ratio]
            bad_invest_WACC_ratio = invest_WACC_ratio[bad_ratio]
            multiple[bad_ratio] = 1 / ((1 - bad_invest_WACC_ratio * bad_WACC_rtn_ratio) / (1 - bad_invest_WACC_ratio))
        return multiple

    def EPV(self, earnings, WACC, growth, dilution):
        EPV_base = (earnings / WACC) * growth
        debt = self.financials.totalDebt()
        cash = self.financials.netCash()
        if isinstance(earnings, pandas.DataFrame):
            EPV_net_debt = EPV_base.sub(debt, axis = "index")
            EPV_net = EPV_net_debt.add(cash, axis = "index")
        else:
            EPV_net = EPV_base - debt + cash
        return EPV_net * dilution

    def per_share(self, values):
        if isinstance(values, pandas.DataFrame):
            value_per_share = values.div(self.financials.numSharesDiluted(), axis = "index")
        else:
            value_per_share = values / self.financials.numSharesDiluted()
        return value_per_share

    def EPV_base(self):
        return self.EPV(self.owner_earnings(), self.WACC_base(), growth = 1, dilution = 1)

    def EPV_adjusted(self):
        return self.EPV(self.owner_earnings(), self.WACC(), growth = self.growth_multiple(), dilution = self.dilution())

    def EPV_growth(self):
        return self.EPV(self.owner_earnings(), self.WACC_base(), growth = self.growth_multiple(), dilution = 1)

    def EPV_levered(self):
        return self.EPV(self.owner_earnings(), self.WACC(), growth = 1, dilution = 1)

    def EPV_cyclic(self):
        return self.EPV(self.ROIC_adjusted_earnings(), self.WACC_base(), growth = 1, dilution = 1)

    def EPV_diluted(self):
        return self.EPV(self.owner_earnings(), self.WACC_base(), growth = 1, dilution = self.dilution())

    def EPV_minimum(self):
        earnings = self.earnings_table()
        earnings = earnings.min(axis = "columns")
        WACC = pandas.DataFrame([self.WACC(), self.WACC_base()]).max()
        growth = self.growth_multiple()
        growth[growth > 1] = 1
        dilution = self.dilution()
        dilution[dilution > 1] = 1
        return self.EPV(earnings, WACC, growth, dilution)

    def EPV_maximum(self):
        earnings = self.earnings_table()
        earnings = earnings.max(axis = "columns")
        WACC = pandas.DataFrame([self.WACC(), self.WACC_base()]).min()
        growth = self.growth_multiple()
        dilution = self.dilution()
        growth[growth < 1] = 1
        dilution[dilution < 1] = 1
        return self.EPV(earnings, WACC, growth, dilution)

    def earnings_table(self):
        table = pandas.DataFrame()
        table["Reported"] = self.financials.reportedIncome()
        table["Adjusted"] = self.owner_earnings()
        table["ROIC"] = self.ROIC_adjusted_earnings()
        table["Trend"] = self.trend_earnings()
        return table.round(1)



class FinanceAnalyst():

    def __init__(self, income, balance, cashflow):
        self.income = income
        self.balance = balance
        self.cashflow = cashflow
        self.operating_cash_pct = 0.015
        self.tax_rate = 0.3
        self.min_rolling_window = 3


    def to_excel(self, writer):
        self.income.to_excel(writer)
        self.balance.to_excel(writer)
        self.cashflow.to_excel(writer)


    def investedCapital(self):
        return self.balance.assets - self.netCash()

    def netCash(self):
        return self.balance.cash - self.operating_cash_pct * self.income.sales

    def totalAssets(self):
        return self.balance.assets

    def totalDebt(self):
        return self.balance.debt

    def shareholdersEquity(self):
        return self.balance.common_equity

    def dividendsCommon(self):
        return self.cashflow.dividends_common

    def DPS_common(self):
        return self.dividendsCommon() / self.numSharesDiluted()

    def diluted_EPS(self):
        return self.income.diluted_EPS

    def capitalInvestmentPct(self):
        return (self.ownerEarnings() - self.dividends_total()) / self.investedCapital()

    def numSharesDiluted(self):
        num_shares = self.income.num_shares_diluted
        return self.series_fillzeroes(num_shares)

    def dilutionGrowth(self):
        num_shares = self.numSharesDiluted()
        yearly_dilution = self.series_pctchange(num_shares)
        return self.rolling_mean(yearly_dilution)

    def reportedIncome(self):
        return self.income.net_to_common
    
    def trendEarnings(self, years = None):     
        return self.series_trend(self.ownerEarnings(), years)

    def ownerEarnings(self):
        EBIT_avg_unusuals = self.EBIT() + self.net_unusuals()
        earnings_after_tax = EBIT_avg_unusuals * (1 - self.tax_rate)
        return earnings_after_tax + self.non_cash_expenses() - self.cash_expenses()

    def ROIC(self):
        return self.ownerEarnings() / self.investedCapital()

    def ROIC_mean(self):
        ROIC = self.ROIC()
        ROIC_trend = self.series_trend(ROIC)
        # Note: The mean is adjusted downwards based on the std dev divided by 
        #  the sqrt of the number of observations.
        #  the obs_adjust is the sqrt of the number of observations. Note that the
        #  number of observations increases.
        obs_adjust = [max(self.min_rolling_window, i) for i in range(1, len(ROIC) + 1)]
        return ROIC_trend - 1.65 * (self.ROIC_std() / obs_adjust)

    def ROIC_std(self):
        ROIC = self.ROIC()
        ROIC_trend = self.series_trend(ROIC)
        ROIC_detrended = ROIC - ROIC_trend
        return self.rolling_std(ROIC_detrended)

    def EBIT(self):
        return self.income.pretax - self.income.net_interest

    def net_unusuals(self):
        unusuals = self.income.unusual_expense - self.income.non_operating_income
        return unusuals - unusuals.mean()

    def non_cash_expenses(self):
        return self.income.DandA

    def cash_expenses(self):
        # Refer 'Cash Expense Analysis - 20160514.xlsx'
        acceptable_range = 0.3
        avg_DandA = pandas.DataFrame([self.income.DandA, self.expended_depreciation()]).mean()
        total_maintenance = self.maintenance_expense()
        total_maint_in_range = ((total_maintenance > self.expended_depreciation() * (1 - acceptable_range)) 
                                & (total_maintenance < self.income.DandA * (1 + acceptable_range)))
        if not any(total_maint_in_range):
            total_maint_ratio = 1
        else:
            total_maint_ratio = total_maintenance[total_maint_in_range].mean() / avg_DandA[total_maint_in_range].mean()

        capital_maintenance = self.capital_injections()
        capital_maint_in_range = ((capital_maintenance > self.expended_depreciation() * (1 - acceptable_range)) 
                                & (capital_maintenance < self.income.DandA * (1 + acceptable_range)))
        if not any(capital_maint_in_range):
            capital_maint_ratio = 1
        else:
            capital_maint_ratio = capital_maintenance[capital_maint_in_range].mean() / avg_DandA[capital_maint_in_range].mean()

        cash_expense = avg_DandA * (total_maint_ratio + capital_maint_ratio) / 2
        return cash_expense

    def maintenance_expense(self):
        return self.PPE_maintenance() + self.intangibles_maintenance() + self.working_capital_requirements()

    def dividend_rate(self):
        return  self.ownerEarnings() / self.dividends_total()

    def dividends_total(self):
        return self.cashflow.dividends_total

    def expended_depreciation(self):
        capital_base_change = self.series_diff(self.capital_base())
        capital_base_change[capital_base_change > 0] = 0
        expended = self.income.DandA + capital_base_change
        expended[expended < 0] = 0
        return expended

    def implied_capex(self):
        capital_base = self.capital_base()
        capital_base_change = self.series_diff(capital_base)
        return capital_base_change + self.income.DandA

    def lease_financing(self):
        # Assumes cash spent on debt over reduction in debt goes 100% towards new leases
        # Positive lease financing indicates increase in leases obtained.
        debt_change = self.series_diff(self.totalDebt())
        return self.cashflow.debt_reduction + debt_change

    def asset_capex(self):
        return self.cashflow.capex_assets

    def capital_injections(self):
        common_equity = self.balance.common_equity
        equity_increase = self.series_diff(common_equity)
        retained_and_added = self.retained_and_added_funds()
        additional_maintenance = retained_and_added - equity_increase
        mean_additional = additional_maintenance[:-1].mean()
        return self.income.DandA + mean_additional

    def retained_and_added_funds(self):
        retained_earnings = self.income.net_to_common - self.cashflow.dividends_total
        stock_par = self.balance.common_carry_value
        added_funds = self.series_diff(stock_par)
        reserves = self.balance.appropriated_reserves
        added_reserves = self.series_diff(reserves)
        return retained_earnings + added_funds + added_reserves

    def PPE_maintenance(self):
        investment = self.lease_financing() + self.asset_capex()
        if investment.sum() == 0:
            net_investment = self.income.depreciation
        else:
            net_investment = investment - self.PPE_change_net_sales()
        return net_investment

    def PPE_change(self):
        PPE_change = self.series_diff(self.balance.PPE)
        PPE_change[PPE_change < 0] = 0
        return PPE_change

    def PPE_change_net_sales(self):
        PPE_change = self.series_diff(self.balance.PPE)
        PPE_sales = self.cashflow.asset_sales
        capped_sales = pandas.concat([PPE_change.abs(), PPE_sales], axis = 1).min(axis = "columns")
        PPE_sales[PPE_change < 0] = capped_sales[PPE_change < 0]
        return PPE_change + PPE_sales

    def intangibles_maintenance(self):
        intangibles_change = self.series_diff(self.balance.other_intangibles)
        intangibles_change[intangibles_change < 0] = 0
        if self.cashflow.capex_other.sum() == 0:
            net_investment = self.income.amortization
        else:
            capex_ex_growth = self.cashflow.capex_other - intangibles_change
            capex_ex_growth[capex_ex_growth < 0] = 0
            intangibles_spend_pct = capex_ex_growth / self.balance.other_intangibles
            intangibles_spend_pct = intangibles_spend_pct[intangibles_spend_pct.notnull()]
            net_investment = self.balance.other_intangibles * intangibles_spend_pct.mean()
        return net_investment

    def working_capital_requirements(self):
        working_capital_change = self.cashflow.change_in_working_capital
        sales_change = self.series_diff(self.income.sales)
        working_capital_ratio = working_capital_change / sales_change
        working_capital_pct = -1 * working_capital_ratio[working_capital_ratio.notnull()].mean()
        return self.income.sales * 0.03 * working_capital_pct

    def growth_capex(self):
        growth_capex = self.series_diff(self.income.sales) * self.cap_sales_ratio()
        growth_capex[growth_capex < 0] = 0
        return growth_capex

    def cap_sales_ratio(self):
        ratios = self.capital_base() / self.income.sales
        return ratios.mean()

    def capital_base(self):
        capital_base = self.balance.PPE + self.balance.intangibles
        return self.series_fillzeroes(capital_base)

    def series_trend(self, series, dates = None):
        if dates is None:
            dates = series.index
        lm = LinearRegression()
        dates_train = np.array([[date.toordinal()] for date in series.index])
        lm.fit(dates_train, series.values)
        dates_predict = [[date.toordinal()] for date in dates]
        return pandas.Series(lm.predict(dates_predict), index = dates)

    def rolling_trend(self, series):

        lm = LinearRegression()
        dates_train = np.array([[date.toordinal()] for date in series.index])

        predicted = pandas.Series(0, index = series.index, dtype = float)
        i = len(dates_train) - self.min_rolling_window

        # The assumption below is that the series is ordered newest first.
        # This is the case for WSJ data but not CMC data.
        predicted.values[i:] = series.values[i:]

        while i >= 0:
            lm.fit(dates_train[i:], series.values[i:])
            predicted.values[i] = lm.predict(dates_train[i])
            i -= 1

        return predicted

    def rolling_mean(self, series):
        # Since the series is stored with years in reverse order, the series needs
        # to be flipped before reverse mean and then flipped back.
        mean = series.rolling(len(series), min_periods = self.min_rolling_window).mean()
        mean = mean.fillna(method = "backfill")
        return mean

    def rolling_std(self, series):
        std = series.rolling(len(series), min_periods = self.min_rolling_window).std()
        std = std.fillna(method = "backfill")
        return std

    def series_diff(self, series):
        delta = series.diff(periods = 1)
        return delta.fillna(method = "backfill")

    def series_fillzeroes(self, series):
        series[series == 0] = np.nan
        series.fillna(method = "backfill")
        return series

    def series_pctchange(self, series):
        change = series.pct_change(periods = 1)
        change.fillna(method = "backfill")
        return change



class Statement():

    def get_row(self, sheet, row_name):
        try:
            row = sheet.loc[row_name]
        except KeyError:
            warnings.warn(row_name + " not found.\n")
            years = sheet.columns
            row = pandas.Series([0.0] * len(years), index = years)
        else:
            row = row.apply(self.convert_to_float)
            row = row * (self.units / 1000)
        return row

    def convert_to_float(self, string):
        string = str(string)
        string = string.replace(",", "")
        if string == "-" or string == "--":
            string = "0.0"
        elif string.startswith("("):
            string = "-" + string.strip("()")
        return float(string)

    def get_units(self, sheet):
        # index name is assumed to be of the form: 
        #    'Fiscal year is <year_start>-<year_end>. All values in <currency> <units>.'
        sheet_label = sheet.index.name
        units_stated = sheet_label.split()[-1]
        units_stated = units_stated.strip(".")
        convert = {"Thousands" : 1000, 
                   "Millions" : 1000000, 
                   "Billions": 1000000000}
        return convert[units_stated]

    def get_fiscal_year_end(self, sheet):
        # index name is assumed to be of the form: 
        #    'Fiscal year is <year_start>-<year_end>. All values in <currency> <units>.'
        sheet_label = sheet.index.name
        fiscal_year = sheet_label.split(".")[0]
        fiscal_year = fiscal_year.split("-")[-1]
        return fiscal_year

    def to_excel(self, writer):
        raise NotImplementedError

    def combine(self, name, annual, interim):
        
        fiscal_month = self.get_fiscal_year_end(annual.income)[0:3]
        annual = getattr(annual, name)
        interim = getattr(interim, name)
        
        half_years = [fiscal_month not in date for date in interim.columns]
        annual.columns = pandas.Index([self.make_datetime(year + "-" + fiscal_month) for year in annual.columns])
        interim.columns = pandas.Index([self.make_datetime(date) for date in interim.columns])

        for row in interim.index:
            if row not in self.annualized:
                try:
                    values = interim.loc[row].apply(self.convert_to_float)
                except ValueError:
                    pass
                else:
                    values = values + values.shift(-1).fillna(method = "pad")
                    values = values.round(1)
                    values = values.apply(str)
                    interim.loc[row] = values

        all_dates = annual.join(interim.loc[:, half_years])
        all_dates = all_dates.sort_index(axis = 1)
        return all_dates

    def make_datetime(self, date_string):
        # Formats for:  Year-mon     WSJ FY      WSJ HY      CMC summary
        date_formats = [ "%Y-%b",     "%Y",    "%d-%b-%Y",     "%m/%y"]

        for format in date_formats:
            try:
                date = datetime.datetime.strptime(date_string, format)
            except ValueError:
                continue
            else:
                return date

        raise ValueError("Unknown date format")


class IncomeStatement(Statement):

    def __init__(self, annual, interim):
        self.annualized = ["Basic Shares Outstanding", "Diluted Shares Outstanding"]
        self.income_sheet = self.combine("income", annual, interim)
        self.units = self.get_units(self.income_sheet)
        

    def to_excel(self, writer):
        self.income_sheet.to_excel(writer, "Income")


    @property
    def sales(self):
        with warnings.catch_warnings(record=False) as w:
            sales =  self.get_row(self.income_sheet, "Sales/Revenue")
        if sales.sum() == 0:
            sales =  self.get_row(self.income_sheet, "Interest Income")
        return sales

    @property
    def DandA(self):
        return self.get_row(self.income_sheet, "Depreciation & Amortization Expense")

    @property
    def depreciation(self):
        return self.get_row(self.income_sheet, "Depreciation")

    @property
    def amortization(self):
        return self.get_row(self.income_sheet, "Amortization of Intangibles")

    @property
    def net_interest(self):
        interest_earned = self.get_row(self.income_sheet, "Non-Operating Interest Income")
        interest_expense = self.get_row(self.income_sheet, "Interest Expense")
        return interest_earned - interest_expense

    @property
    def pretax(self):
        return self.get_row(self.income_sheet, "Pretax Income")

    @property
    def num_shares_diluted(self):
        return self.get_row(self.income_sheet, "Diluted Shares Outstanding")

    @property
    def COGS(self):
        return self.get_row(self.income_sheet, "COGS excluding D&A")

    @property
    def SGA(self):
        with warnings.catch_warnings(record=False) as w:
            SGA =  self.get_row(self.income_sheet, "SG&A Expense")
        if SGA.sum() == 0:
            SGA =  self.get_row(self.income_sheet, "Selling, General & Admin. Expenses")
        return SGA

    @property
    def net_to_common(self):
        return self.get_row(self.income_sheet, "Net Income Available to Common")

    @property
    def unusual_expense(self):
        return self.get_row(self.income_sheet, "Unusual Expense")

    @property
    def non_operating_income(self):
        return self.get_row(self.income_sheet, "Non Operating Income/Expense")

    @property
    def diluted_EPS(self):
        return (self.net_to_common / self.num_shares_diluted).round(3)



class BalanceSheet(Statement):

    def __init__(self, annual, interim):
        self.annualized = annual.assets.index.tolist() + annual.liabilities.index.tolist()
        self.asset_sheet = self.combine("assets", annual, interim)
        self.liabilities = self.combine("liabilities", annual, interim)
        self.units = self.get_units(self.asset_sheet)

    def to_excel(self, writer):
        self.asset_sheet.to_excel(writer, "Balance")
        self.liabilities.to_excel(writer, "Balance", startrow = len(self.asset_sheet) + 1)


    @property
    def PPE(self):
        return self.get_row(self.asset_sheet, "Net Property, Plant & Equipment")

    @property
    def intangibles(self):
        return self.get_row(self.asset_sheet, "Intangible Assets")

    @property
    def other_intangibles(self):
        return self.get_row(self.asset_sheet, "Net Other Intangibles")

    @property
    def assets(self):
        return self.get_row(self.asset_sheet, "Total Assets")

    @property
    def cash(self):
        return self.get_row(self.asset_sheet, "Cash & Short Term Investments")

    @property
    def debt(self):
        long_debt = self.get_row(self.liabilities, "Long-Term Debt")
        short_debt = self.get_row(self.liabilities, "ST Debt & Current Portion LT Debt")
        return long_debt + short_debt

    @property
    def common_equity(self):
        return self.get_row(self.liabilities, "Common Equity (Total)")

    @property
    def common_carry_value(self):
        return self.get_row(self.liabilities, "Common Stock Par/Carry Value")

    @property
    def appropriated_reserves(self):
        return self.get_row(self.liabilities, "Other Appropriated Reserves")



class CashflowStatement(Statement):

    def __init__(self, annual, interim):
        self.annualized = ["None"]
        self.operating = self.combine("operating", annual, interim)
        self.investing = self.combine("investing", annual, interim)
        self.financing = self.combine("financing", annual, interim)
        self.units = self.get_units(self.operating)


    def to_excel(self, writer):
        self.operating.to_excel(writer, "Cashflow")
        self.investing.to_excel(writer, "Cashflow", startrow = len(self.operating) + 1)
        self.financing.to_excel(writer, "Cashflow", startrow = len(self.operating) + len(self.investing) + 2)


    @property
    def capex_assets(self):
        capex = self.get_row(self.investing, "Capital Expenditures (Fixed Assets)")
        return -1 * capex

    @property
    def capex_other(self):
        capex = self.get_row(self.investing, "Capital Expenditures (Other Assets)")
        return -1 * capex

    @property
    def dividends_total(self):
        dividends = self.get_row(self.financing, "Cash Dividends Paid - Total")
        return -1 * dividends

    @property
    def dividends_common(self):
        dividends = self.get_row(self.financing, "Common Dividends")
        return -1 * dividends

    @property
    def asset_sales(self):
        return self.get_row(self.investing, "Sale of Fixed Assets & Businesses")

    @property
    def debt_reduction(self):
        cash_change_from_debt = self.get_row(self.financing, "Issuance/Reduction of Debt, Net")
        return -1 * cash_change_from_debt

    @property
    def change_in_working_capital(self):
        return self.get_row(self.operating, "Changes in Working Capital")



class PriceAnalyser():

    def __init__(self, prices):
        self.prices = prices

    def prices_FY(self, date):
        FY_start = datetime.date(date.year - 1, 7, 1)
        FY_end = datetime.date(date.year, 6, 30)
        return self.prices[FY_start:FY_end]

    def latest_price(self):
        return self.prices["Close"].iloc[-1]

    def average_price(self, date):
        prices = self.prices_FY(date)
        return prices["Close"].mean()

    def high_price(self, date):
        prices = self.prices_FY(date)
        return prices["High"].max()

    def low_price(self, date):
        prices = self.prices_FY(date)
        return prices["Low"].min()

    def stddev_price(self, date):
        prices = self.prices_FY(date)
        return prices["Close"].std()

    def price_table(self, dates):
        headings = ["Average" , "High", "Low" , "Std Dev"]
        table = pandas.DataFrame(None, index = dates, columns = headings, dtype = float)
        for date in dates:
            table["Average"][date] = self.average_price(date)
            table["High"][date] = self.high_price(date)
            table["Low"][date] = self.low_price(date)
            table["Std Dev"][date] = self.stddev_price(date)
        return table

    def price_appreciation(self, ticker, valuations, type, periods):
        '''
        This method calculates the change in price for a given valuation period.
        Inputs:
        ticker - ticker of the stock to analyse
        valuations - the Valuations storage resource object
        type - the type of valuation to investigate (e.g. Base, Min, Adjusted).
        periods - integer list of months to investigate (e.g. appreciate after 3, 6, 12 months)
        '''
        
        columns = ["ticker", "value", "start"] + [str(period) for period in periods]
        appreciation = pandas.DataFrame(np.nan, index = valuations.index, columns = columns)
        for date in valuations[type].index:
            valuation = valuations[type][date]
            value_date = datetime.date(date.year, date.month, 20)
            start = value_date + relativedelta(months = 2)
            end = value_date + relativedelta(months = 8)
            start_price = self.prices["Close"][start:end].quantile(0.2)

            appreciation.loc[date, "ticker"] = ticker
            appreciation.loc[date, "value"] = valuation
            appreciation.loc[date, "start"] = start_price

            for period in periods:
                period_start = start + relativedelta(months = period)
                period_end = end + relativedelta(months = period)
                appreciation.loc[date, str(period)] = self.prices["Close"][period_start:period_end].median()

        return appreciation

    def plotPriceMovesVsValue(self, price_moves, periods, segments = 6, upper_ratio = 10, lower_ratio = 0, ylim = [-0.5, 1]):

        pct_moves = price_moves.copy()
        pct_moves["start"] = price_moves["start"] / price_moves["value"]
        pct_moves.loc[:, periods] = price_moves.loc[:, periods].div(price_moves["start"], axis = 'index') - 1
        pct_moves = pct_moves[pct_moves["start"] > lower_ratio]
        pct_moves = pct_moves[pct_moves["start"] < upper_ratio]

        cut_factors = pandas.qcut(pct_moves["start"], segments)
        xlabels = cut_factors.cat.categories.tolist()
        figure = pct_moves.boxplot(column = periods, by = cut_factors)
        if isinstance(figure, np.ndarray):
            for ax in figure.reshape(-1):
                ax.set_ylim(ylim)
                ax.set_xlabel("")
                ax.set_xticklabels(xlabels, rotation = 45, fontsize = 10, ha = "right")
                ax.set_title("")
                fig = ax.get_figure()
                fig.suptitle("")
        else:
            plt.ylim(ylim)
            plt.xlabel("")
            plt.xticks(rotation = 45, fontsize = 10, ha = "right")
            fig = figure.get_figure()
            fig.suptitle("")
        plt.title("")
        plt.subplots_adjust(bottom = 0.15)
        plt.show()




