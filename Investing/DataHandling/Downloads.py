import requests
import os
import pandas
import datetime
from pandas_datareader import data as pd_data
from pandas_datareader import base as pd_base
from bs4 import BeautifulSoup


class WSJdownloader():

    def __init__(self, store):
        self.page_root = "http://quotes.wsj.com/AU/XASX/"
        self.store = store
        self.stock_pages = {"overview" : "", 
                            "financials" : "/financials", 
                            "income" : "/financials/annual/income-statement", 
                            "balance" : "/financials/annual/balance-sheet", 
                            "cashflow" : "/financials/annual/cash-flow"}
        

    def get_address(self, ticker, type):
        return  self.page_root + ticker + self.stock_pages[type]

    def download_pages(self, ticker):
        for type in self.stock_pages.keys():
            try:
                overview = requests.get(self.get_address(ticker, type))
                self.save_page(overview.content, self.store.html(ticker, type))
            except requests.HTTPError:
                print("Problem downloading: " + ticker + " " + type)
            
    def save_page(self, page, file_path):
        self.store.check_directory(file_path)
        with open(file_path, "w") as file:
            file.write(page)

    def download_all(self, tickers):
        for ticker in tickers:
            self.download_pages(ticker)


class StockFinancialsResource():

    def __init__(self, store):
        self.store = store


    def financialsToPickle(self, tickers):
        scraper = WSJscraper(self.store)
        tables = ["income", "assets", "liabilities", "operating", "investing", "financing"]
        for ticker in tickers:
            for table in tables:
                method_name = "read_" + table + "_table"
                html_reader = getattr(scraper, method_name)
                try:
                    finance_sheet = html_reader(ticker)
                except InsufficientDataError:
                    print(" ".join([ticker, table]))
                except Exception as E:
                    print(type(E), E.message)
                else:
                    file_path = self.store.stock_pickle(ticker, table)
                    self.store.check_directory(file_path)
                    finance_sheet.to_pickle(file_path)

    def getFinancials(self, ticker, type):
        file_path = self.store.stock_pickle(ticker, type)
        try:
            finance_sheet = pandas.read_pickle(file_path)
        except:
            html_reader = getattr(WSJscraper(self.store), "read_" + type + "_table")
            finance_sheet = html_reader(ticker)
        return finance_sheet


class XLSio():

    def __init__(self, store):
        self.store = store

    # NOTE: This is currently working on the assumption that the workbook only
    # contains one worksheet.
    # Takes inputs from ASX Listed Companies downloaded from ASX.com.au
    def loadWorkbook(self, name):
        if name is "ASXListedCompanies":
            header = 2
        else:
            header = 0
        table = pandas.read_excel(self.store.excel(name), header = header)
        table.index = table.pop("ASX code")
        self.table = table

    def getHeader(self):
        return self.table.columns.tolist()

    def getTickers(self):
        return self.table.index.tolist()

    def updateTable(self, new_data):
        new_table = pandas.DataFrame.from_dict(new_data, orient = "index")
        self.table = self.table.join(new_table)

    def saveAs(self, filename):
        self.table.to_excel(self.store.excel(filename), sheet_name = "Stock table")


class Storage():
    
    def __init__(self, root_folder = "D:\\Investing\\Data\\"):
        self.root = root_folder

    def excel(self, filename):
        return os.path.join(self.root, filename + ".xlsx")

    def html(self, ticker, type):
        return os.path.join(self.root, ticker, ticker + type + ".html")

    def yahoo(self, ticker):
        return os.path.join(self.root, ticker, ticker + "prices.pkl")

    def stock_pickle(self, ticker, type):
        return os.path.join(self.root, ticker, "pickles", type + ".pkl")

    def check_directory(self, file_path):
        path = os.path.dirname(file_path)
        if not os.path.exists(path):
            os.makedirs(path)


class YahooDataDownloader():
    '''
    Uses the Pandas data functionality to download data and handle local storage.
    '''
    def __init__(self, storage):
        self.store = storage
        
    def get(self, ticker, start = None, end = None):
        if start is None:
            start = datetime.date(2010, 01, 01)
        if end is None:
            end = datetime.date.today()
        self.ticker = ticker
        return pd_data.get_data_yahoo(ticker + ".AX", start, end)

    def download_all(self, tickers = None, start = None, end = None):
        if tickers is None:
            xls = XLSio(self.store)
            xls.loadWorkbook("ASXListedCompanies")
            tickers = xls.getTickers()
        for ticker in tickers:
            try:
                prices = self.get(ticker, start, end)
            except pd_base.RemoteDataError:
                print("Problem downloading: " + ticker)
            else:
                file_path = self.store.yahoo(ticker)
                self.store.check_directory(file_path)
                prices.to_pickle(file_path)

    def current_price(self, ticker):
        ticker = ticker + ".AX"
        quote = pd_data.get_quote_yahoo(ticker)
        return quote["last"][ticker]


class WSJscraper():

    def __init__(self, store):
        self.store = store


    def load_overview(self, ticker):
        overview = self.store.html(ticker, "overview")
        with open(overview, 'r') as page:
            self.overview = BeautifulSoup(page, "lxml")

    def keyStockData(self):
        key_data_table = self.overview.find(id = "cr_keystock_drawer")
        key_data_table = key_data_table.find("div")
        table_entries = key_data_table.find_all("li")
        labels = []
        values = []
        for entry in table_entries:
            label = str(entry.h5.text)
            labels.append(label)
            
            # Clean up tag string content. Note some tags may have leading white space
            # before a child tag and so all text is retrieved (ignoring children) and
            # combined, before removing unwanted characters.
            value = ''.join(entry.span.find_all(text = True, recursive = False))
            value = value.strip()
            value = value.replace("%", "")
            value = value.replace("M", "")
            
            try:
                value = float(value)
            except ValueError:
                value = str(value)

            values.append(value)

        return dict(zip(labels, values))

    def read_statement_table(self, ticker, type, contains):
        try:
            table = pandas.read_html(self.store.html(ticker, type), match = contains, index_col = 0)[0]
        except ValueError as E:
            raise MissingStatementEntryError(E.message)
        headings = table.columns.tolist()
        # Delete empty columns after final year
        for heading in headings[headings.index("5-year trend"):]:
            del table[heading]
        # Delete rows starting with 'nan'
        non_nans = [not isinstance(row_label, float) for row_label in table.index]
        table = table.loc[non_nans]
        self.check_years(table.columns.tolist())
        return table

    def check_years(self, years):
        if not all(['20' in year[0:2] for year in years]):
            raise InsufficientDataError("Empty report years")

    def read_income_table(self, ticker):
        return self.read_statement_table(ticker, "income", "Sales/Revenue")

    def read_assets_table(self, ticker):
        return self.read_statement_table(ticker, "balance", "Cash & Short Term Investments")

    def read_liabilities_table(self, ticker):
        return self.read_statement_table(ticker, "balance", "ST Debt & Current Portion LT Debt")

    def read_operating_table(self, ticker):
        return self.read_statement_table(ticker, "cashflow", "Net Operating Cash Flow")

    def read_investing_table(self, ticker):
        return self.read_statement_table(ticker, "cashflow", "Capital Expenditures")

    def read_financing_table(self, ticker):
        return self.read_statement_table(ticker, "cashflow", "Cash Dividends Paid - Total")


class InsufficientDataError(IOError):
    pass

class MissingStatementEntryError(IOError):
    pass

