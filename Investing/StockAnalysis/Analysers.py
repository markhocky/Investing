import pandas
import math
import datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from DataHandling.Downloads import Storage, StockFinancialsResource, WSJscraper, XLSio, YahooDataDownloader, MissingStatementEntryError, InsufficientDataError

from multiprocessing import Process, Queue, Pool


# TODO - REFINEMENTS
# Growth multiple doesn't seem to work well. Sometimes is higher than appears justified.
# Add stock blacklist to avoid cycling through those with errors every time.

# TODO - REFACTORING
# Move series manipulations (trend, diff etc) to object derived from pandas.Series
# Statement can create this object on get_row

# TODO - FIXES
# Improve error handling in storeValuationSummaryBrief
# Apply appropriate corrections when not all years have reported values.
# Apply to: Capital Base, Total Assets
#
# Errors:
# Account for USD reported accounts e.g. RMD
#
# Aliases:
# "Cash & Short Term Investments" - "Cash Only" (AMP, IAG, ISU, MPL, QBE, SUN)
# Handle statements for financial companies with no "Sales/Revenue":
# ANZ, ABA, BOQ, BEN, CBA, GMY, IQE, KMD, KNH, MQG, WBC
#
# Data not available: KMD, KNH, MEZ, MPP, NEC, NPX, SNC, SFL, TGZ

# TODO - FEATURES
# Add Book Valuations.
# DONE - Parallel processing of valuations.
# DONE - Update valuations based on latest prices only.
# Analysis of financial stocks - Banks and Insurance.
# Add data from analyst forecast earnings.
# Add data from competitors and industry.


def retrieveOverviewData(storage_dir, headings = None):
    store = Storage(storage_dir)
    xls = XLSio(store)
    scraper = WSJscraper(store)
    xls.loadWorkbook("ASXListedCompanies")
    tickers = xls.getTickers()
    new_data = {}
    for ticker in tickers:
        scraper.load_overview(ticker)
        try:
            new_data[ticker] = scraper.keyStockData()
        except Exception:
            print("Problem with: " + ticker)
    xls.updateTable(new_data)
    xls.saveAs("StockSummary")

def storeValuationSummaryBrief(tickers = None):
    store = Storage()
    if tickers is None:
        xls = XLSio(store)
        xls.loadWorkbook("StockSummary")
        xls.table = xls.table[xls.table["P/E Ratio (TTM)"].notnull()]
        tickers = xls.getTickers()
    summary = {}
    count = 1.0
    errors = {}
    print("Assessing " + str(len(tickers)) + " companies")
    for ticker in tickers:
        try:
            reporter = Reporter(ticker)
            summary[ticker] = reporter.oneLineValuation()
        except MissingStatementEntryError as E:
            errors[ticker] = E.message
        except InsufficientDataError as E:
            errors[ticker] = E.message
        except Exception as E:
            errors[ticker] = E.message
        if count % max(len(tickers) / 4, 1) == 0:
            pct_complete = round(100.0 * count / len(tickers))
            print(str(pct_complete) + "% complete")    
        count += 1
    
    index = summary.items()[0][1].index        
    summary = pandas.DataFrame(summary, index = index).T
    summary.to_excel(store.excel("ValuationSummary"))
    print(str(len(errors)) + " Failed")
    print(str(len(summary)) + " Succeeded")
    return errors

def buildFinancialAnalyst(ticker, storage_dir = "D:\\Investing\\Data"):
    resource = StockFinancialsResource(Storage(storage_dir))
    income = IncomeStatement(resource.getFinancials(ticker, "income"))
    balance = BalanceSheet(resource.getFinancials(ticker, "assets"), resource.getFinancials(ticker, "liabilities"))
    cashflow = CashflowStatement(resource.getFinancials(ticker, "operating"), resource.getFinancials(ticker, "investing"), resource.getFinancials(ticker, "financing"))
    return FinanceAnalyst(income, balance, cashflow)

def buildPriceAnalyser(ticker, storage_dir = "D:\\Investing\\Data"):
    store = Storage(storage_dir)
    prices = pandas.read_pickle(store.yahoo(ticker))
    return PriceAnalyser(prices) 

def saveAnalysisToExcel(ticker):
    results = Reporter(ticker)
    store = Storage()
    writer = pandas.ExcelWriter(store.excel(ticker + "analysis"))
    results.summaryTable().to_excel(writer, "Summary")
    results.financials.income.income_sheet.to_excel(writer, "Income")
    assets = results.financials.balance.asset_sheet
    liabilities = results.financials.balance.liabilities
    assets.to_excel(writer, "Balance")
    liabilities.to_excel(writer, "Balance", startrow = len(assets) + 1)
    operating = results.financials.cashflow.operating
    investing = results.financials.cashflow.investing
    financing = results.financials.cashflow.financing
    operating.to_excel(writer, "Cashflow")
    investing.to_excel(writer, "Cashflow", startrow = len(operating) + 1)
    financing.to_excel(writer, "Cashflow", startrow = len(operating) + len(investing) + 2)
    writer.save()

def getOneLineSummary(ticker):
    try:
        reporter = Reporter(ticker)
        result = reporter.oneLineValuation()
    except MissingStatementEntryError as E:
        result = "Error valuing {0}: {1}".format(ticker, E.message)
    except InsufficientDataError as E:
        result = "Error valuing {0}: {1}".format(ticker, E.message)
    except Exception as E:
        result = "Error valuing {0}: {1}".format(ticker, E.message)
    
    return result

def parallelSummary(tickers = None):
    store = Storage()
    if tickers is None:
        xls = XLSio(store)
        xls.loadWorkbook("StockSummary")
        xls.table = xls.table[xls.table["P/E Ratio (TTM)"].notnull()]
        tickers = xls.getTickers()
    
    pool = Pool(processes = 8)
    all_results = pool.map(getOneLineSummary, tickers)
    errors = []
    results = []

    for result in all_results:
        if type(result) is pandas.Series:
            results.append(result)
        else:
            errors.append(result)

    results = pandas.concat(results, axis = 1).T

    results.to_excel(store.excel("ValuationSummary"))
    print(str(len(errors)) + " Failed")
    print(str(len(results)) + " Succeeded")

    return errors

def updatePrices(date = None):
    store = Storage()
    if date is None:
        valuations = store.list_files(store.excel(), "Valuation")
        dates = [val[-13:-5] for val in valuations]
        date = max(dates)
    summary = pandas.read_excel(store.excel("ValuationSummary" + date), index_col = 0)
    yahoo = YahooDataDownloader("")
    new_prices = pandas.Series([yahoo.current_price(ticker) for ticker in summary.index], index = summary.index)
    old_prices = summary["Current Price"]
    values = summary.loc[:, "Adjusted Value":"Dilution Value"]
    new_values = (values + 1).multiply(old_prices / new_prices, axis = "rows") - 1
    summary.loc[:, "Adjusted Value":"Dilution Value"] = new_values
    summary["Current Price"] = new_prices
    summary["Current PE"] = summary["Current EPS"] / summary["Current Price"]
    summary.to_excel(store.excel("ValuationSummary20160622"))
    return summary



class Reporter():
    
    def __init__(self, ticker, analyse = True):
        self.ticker = ticker
        self._index = None
        self._maintenance = None
        self._earnings = None
        self._EPV = None
        self._prices = None
        self._values = None
        if analyse:
            self.financials = buildFinancialAnalyst(ticker)
            self.EPVanalyser = EPVanalyser(self.financials)
            self.price_analyser = buildPriceAnalyser(ticker)
            self.current_price = YahooDataDownloader("").current_price(ticker)


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
        table["Mean Return (%)"] = self.pad_row(self.format_pct(self.EPVanalyser.ROIC_mean()))
        table["Std Dev Return (%)"] = self.pad_row(self.format_pct(self.EPVanalyser.ROIC_std()))
        table["opt F"] = self.pad_row(self.format_ratio(self.EPVanalyser.optF()))
        table["act F"] = self.pad_row(self.format_ratio(self.EPVanalyser.actF()))
        table["WACC base (%)"] = self.pad_row(self.format_pct(self.EPVanalyser.WACC_base()))
        table["WACC adj. (%)"] = self.pad_row(self.format_pct(self.EPVanalyser.WACC()))
        table["Growth mult."] = self.pad_row(self.format_ratio(self.EPVanalyser.growth_multiple().iloc[0]))
        table["Dilution (%)"] = self.pad_row(self.format_pct(self.EPVanalyser.dilution()))
        return table

    def shareValue_table(self):
        values = self.EPVanalyser.per_share(self.EPV)
        return values.round(3)

    def prices_table(self):
        prices = self.price_analyser.price_table(self.index)
        return prices.round(3)

    def valuation_pct(self):
        valuation = self.share_values.iloc[0]
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
        summary["Current EPS"] = self.EPS[0]
        summary["Current Div"] = self.DPS[0]
        summary["Average Div"] = self.DPS.mean()
        summary = summary.append(self.valuationMetrics_table().iloc[0])
        valuation = self.valuation_pct().round(3)
        valuation = self.append_label(valuation, "Value")
        summary = summary.append(valuation)
        summary = summary.append(self.PE_ranges().round(1))
        summary.name = self.ticker
        return summary

    def PE_ranges(self):
        PE_summary = pandas.Series(dtype = float)
        PE_summary["Current PE"] = self.current_price / self.EPS[0]
        PE_current = self.PE.iloc[0]
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
        growth_multiple = self.EPVanalyser.growth_multiple().iloc[0]
        franchise_factor = (mean_rtn - capital_cost) / (capital_cost * mean_rtn)
        theoretical_PE = (1 / capital_cost) + franchise_factor * growth_multiple
        book_value = self.EPVanalyser.book_value().iloc[0]
        franchise_value = (1 / capital_cost) * (mean_rtn - capital_cost) * (growth_multiple * book_value)

        table["Mean Return (%)"] = self.pad_row(self.format_pct(mean_rtn))
        table["Franchise Return (%)"] = self.pad_row(self.format_pct(mean_rtn))
        table["WACC base (%)"] = self.pad_row(self.format_pct(self.EPVanalyser.WACC_base()))
        table["WACC adj. (%)"] = self.pad_row(self.format_pct(capital_cost))
        table["Growth mult."] = self.pad_row(self.format_ratio(growth_multiple))
        table["Franchise Factor (FF)"] = self.pad_row(self.format_ratio(franchise_factor))
        table["Theoretical PE"] = self.pad_row(self.format_ratio(theoretical_PE))
        table["Book Value ($)"] = self.pad_row(self.format_ratio(book_value))
        table["Franchise Value ($)"] = self.pad_row(self.format_ratio(franchise_value))

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
        return pandas.Series([value] + ["-"] * (len(self.index) - 1), index = self.index)

    def format_pct(self, number):
        return round(100.0 * number, 2)

    def format_ratio(self, number):
        return round(number, 2)

    def valueHistogram(self):
        valuation = self.valuation_pct()
        valuation.pop("All +ve")
        ax = valuation.plot(kind = "bar")
        ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in ax.get_yticks()])
        plt.xticks(rotation = 45)
        plt.subplots_adjust(bottom = 0.15)
        plt.show()

    def valueVSprice(self):
        values = self.share_values[::-1]
        xlabels = values.index.tolist()
        prices = self.prices[::-1]
        value_points = plt.plot(values, 'o')
        ax = plt.axes()
        ax.set_yticklabels(['$%s' % float('%.3g' % x) for x in ax.get_yticks()])
        plt.legend(value_points, values.columns.tolist())
        plt.plot(prices["Average"], 'r')
        plt.plot(prices["High"], "r--")
        plt.plot(prices["Low"], "r--")
        plt.plot(len(xlabels) - 1, self.current_price, "rs")
        plt.xlim(-0.5, len(xlabels) - 0.5)
        plt.xticks(range(len(xlabels)), xlabels)
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
        return self.owner_earnings() / self.financials.investedCapital()

    def ROIC_mean(self):
        ROIC = self.ROIC()
        ROIC_trend = self.financials.series_trend(ROIC)
        ROIC_current = ROIC_trend.iloc[0]
        return ROIC_current - 1.65 * (self.ROIC_std() / (5.0 ** 0.5))

    def ROIC_std(self):
        ROIC = self.ROIC()
        ROIC_trend = self.financials.series_trend(ROIC)
        return (ROIC - ROIC_trend).std()

    def optF_theoretical(self):
        min_denom = 0.001
        adj_frac = 1.0 / 6.0
        min_F = 0.001
        mean = self.ROIC_mean()
        std = self.ROIC_std()
        return max(adj_frac * mean / max(std ** 2 - mean ** 2, min_denom), min_F)

    def optF_simplified(self):
        mean = self.ROIC_mean()
        std = self.ROIC_std()
        adj_frac = 1.0 / 6.0
        min_F = 0.001
        return max(adj_frac * mean / (std ** 2), min_F)

    def optF_max_allowable_loss(self):
        mean = self.ROIC_mean()
        std = self.ROIC_std()
        three_sigma_loss = mean - 3 * std
        if three_sigma_loss >= 0:
            optF = 10.0
        else:
            optF = -1 / three_sigma_loss
        return optF

    def optF(self):
        return min([self.optF_theoretical(), self.optF_simplified(), self.optF_max_allowable_loss()])

    def actF(self):
        # Actual leverage is calculated as if all free cash is used to pay down debt.
        assets = self.financials.totalAssets()[0]
        debt = self.financials.totalDebt()[0]
        net_cash = self.financials.netCash()[0]
        return (assets - net_cash) / (assets - debt)

    def WACC(self, equity_cost = None):
        if equity_cost is None:
            equity_cost = self.equity_cost()
        optF = self.optF()
        return equity_cost * (1.0 / optF) + self.debt_cost * ((optF - 1.0) / optF)

    def WACC_base(self):
        equity_cost = self.equity_base_cost + self.equity_premium
        return self.WACC(equity_cost)

    def loss_probability(self, optF):
        mean_rtn = self.ROIC_mean()
        std_rtn = self.ROIC_std()
        try:
            probability = math.exp((-2.0 * optF * mean_rtn * self.acceptable_DD) / (optF * std_rtn) ** 2)
        except OverflowError:
            probability = 1
        return probability

    def equity_cost(self):
        prob_opt = self.loss_probability(self.optF())
        prob_act = self.loss_probability(self.actF())
        if prob_opt > 0.05:
            leverage_ratio = 1
        else:
            leverage_ratio = prob_act / prob_opt
        return self.equity_base_cost + leverage_ratio * self.equity_premium

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
        if isinstance(invest_WACC_ratio, pandas.Series):
            invest_WACC_ratio[invest_WACC_ratio > 0.75] = 0.75
            invest_WACC_ratio[(WACC > mean_rtn) & (invest_WACC_ratio < 0)] = 0.75
        else:
            invest_WACC_ratio = min(invest_WACC_ratio, 0.75)
            if WACC > mean_rtn and invest_WACC_ratio < 0:
                invest_WACC_ratio = 0.75
        # Ratio of WACC to Return adjusted for neg return conditions.
        if mean_rtn == 0:
            WACC_rtn_ratio = 2
        elif mean_rtn > 0:
            WACC_rtn_ratio = WACC / mean_rtn
        else:
            WACC_rtn_ratio = max(abs((WACC - mean_rtn) / mean_rtn), 1.5)

        if WACC_rtn_ratio > 1:
            # If cost of capital (WACC) is greater than mean_rtn, then growth is bad.
            # We want to return a multiple less than 1.
            WACC_rtn_ratio = 1 / WACC_rtn_ratio
            multiple = 1 / ((1 - invest_WACC_ratio * WACC_rtn_ratio) / (1 - invest_WACC_ratio))
        else:
            # The normal case
            multiple = (1 - invest_WACC_ratio * WACC_rtn_ratio) / (1 - invest_WACC_ratio)

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
        WACC = max([self.WACC(), self.WACC_base()])
        growth_multiple = self.growth_multiple()
        if not isinstance(growth_multiple, float):
            growth_multiple = growth_multiple[0]
        growth = min(growth_multiple, 1)
        dilution = min(1, self.dilution())
        return self.EPV(earnings, WACC, growth, dilution)

    def EPV_maximum(self):
        earnings = self.earnings_table()
        earnings = earnings.max(axis = "columns")
        WACC = min([self.WACC(), self.WACC_base()])
        growth_multiple = self.growth_multiple()
        if not isinstance(growth_multiple, float):
            growth_multiple = growth_multiple[0]
        growth = max(growth_multiple, 1)
        dilution = max(1, self.dilution())
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
        yearly_shares_growth = num_shares.pct_change(periods = -1)
        return yearly_shares_growth.mean()

    def reportedIncome(self):
        return self.income.net_to_common
    
    def trendEarnings(self, years = None):     
        return self.series_trend(self.ownerEarnings(), years)

    def ownerEarnings(self):
        EBIT_avg_unusuals = self.EBIT() + self.net_unusuals()
        earnings_after_tax = EBIT_avg_unusuals * (1 - self.tax_rate)
        return earnings_after_tax + self.non_cash_expenses() - self.cash_expenses()

    def EBIT(self):
        return self.income.pretax - self.income.net_interest

    def net_unusuals(self):
        unusuals = self.income.unusual_expense - self.income.non_operating_income
        return unusuals - unusuals.mean()

    def non_cash_expenses(self):
        return self.income.DandA

    def cash_expenses(self):
        # Refer Cash Expense Analysis - 20160514.xlsx
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

    def series_trend(self, series, years = None):
        if years is None:
            years = series.index
        lm = LinearRegression()
        years_train = np.array([[float(year)] for year in series.index])
        lm.fit(years_train, series.values)
        years_predict = [[float(year)] for year in years]
        return pandas.Series(lm.predict(years_predict), index = years)

    def series_diff(self, series):
        delta = series.diff(periods = -1)
        return delta.fillna(method = "pad")

    def series_fillzeroes(self, series):
        series[series == 0] = np.nan
        series.fillna(method = "pad")
        return series



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
        if string == "-":
            string = "0.0"
        elif string.startswith("("):
            string = "-" + string.strip("()")
        return float(string)

    def get_units(self, sheet):
        sheet_label = sheet.index.name
        units_stated = sheet_label.split()[-1]
        units_stated = units_stated.strip(".")
        convert = {"Thousands" : 1000, 
                   "Millions" : 1000000, 
                   "Billions": 1000000000}
        return convert[units_stated]



class IncomeStatement(Statement):

    def __init__(self, income_sheet):
        self.income_sheet = income_sheet
        self.units = self.get_units(income_sheet)


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

    def __init__(self, assets, liabilities):
        self.asset_sheet = assets
        self.liabilities = liabilities
        self.units = self.get_units(assets)


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

    def __init__(self, operating, investing, financing):
        self.operating = operating
        self.investing = investing
        self.financing = financing
        self.units = self.get_units(operating)


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

    def prices_FY(self, year):
        FY_start = datetime.date(year - 1, 07, 01)
        FY_end = datetime.date(year, 06, 30)
        return self.prices[FY_start:FY_end]

    def average_price(self, year):
        prices = self.prices_FY(year)
        return prices["Close"].mean()

    def high_price(self, year):
        prices = self.prices_FY(year)
        return prices["High"].max()

    def low_price(self, year):
        prices = self.prices_FY(year)
        return prices["Low"].min()

    def stddev_price(self, year):
        prices = self.prices_FY(year)
        return prices["Close"].std()

    def price_table(self, years):
        headings = ["Average" , "High", "Low" , "Std Dev"]
        table = pandas.DataFrame(None, index = years, columns = headings, dtype = float)
        for year in years:
            table["Average"][year] = self.average_price(int(year))
            table["High"][year] = self.high_price(int(year))
            table["Low"][year] = self.low_price(int(year))
            table["Std Dev"][year] = self.stddev_price(int(year))
        return table



