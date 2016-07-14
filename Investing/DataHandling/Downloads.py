import requests
import os
import shutil
import pandas
import datetime
import pickle
from pandas_datareader import data as pd_data
from pandas_datareader import base as pd_base
from bs4 import BeautifulSoup


class WebDownloader():
    
    def __init__(self):
        self.store = Storage()
        self.WSJ = WSJinternet()
        self.Yahoo = YahooDataDownloader()


    def updateFinancials(self, tickers, period):
        for ticker in tickers:
            
            financials_template = Financials(ticker, period)
            try:
                financials = self.store.load(financials_template)
            except IOError:
                financials = financials_template

            try:
                new_financials = self.WSJ.getFinancials(ticker, period)
            except Exception as e:
                print(e.message + " - problem with " + ticker)
            else:
                financials.merge(new_financials)
                self.store.save(financials)

    def updatePriceHistory(self, tickers, start = None):
        for ticker in tickers:
            price_history = PriceHistory(ticker)
            try:
                prices_history.prices = self.Yahoo.priceHistory(ticker, start)
            except Exception as e:
                print(e.message + " - problem getting " + ticker)
            else:
                self.store.save(prices)

    def currentPrice(self, ticker):
        return self.Yahoo.currentPrice(ticker)

    def all_tickers(self):
        return [ticker for ticker in os.listdir(self.store.root) if "." not in ticker]



class StorageResource():

    def selectFolder(self, store):
        raise NotImplementedError

    def filename(self):
        raise NotImplementedError

    def loadFrom(self, file_path):
        raise NotImplementedError

    def saveTo(self, file_path):
        raise NotImplementedError


class Financials(StorageResource):

    def __init__(self, ticker, period):
        self.ticker = ticker
        self.period = period.lower()
        self.statements = {}

    def merge(self, other):
        self.confirm_match(other.ticker, other.period)
        for sheet in other.statements:
            try:
                existing_sheet = self.statements[sheet]
            except KeyError:
                self.statements[sheet] = other.statements[sheet]
            else:
                new_sheet = other.statements[sheet]
                for table in new_sheet:
                    existing_table = existing_sheet[table]
                    new_table = new_sheet[table]
                    joined_table = self.merge_columns(existing_table, new_table)
                    self.statements[sheet][table] = joined_table

    def merge_columns(self, existing, new):
        existing_years = existing.columns.tolist()
        new_years = new.columns.tolist()
        append_years = [year not in new_years for year in existing_years]
        return pandas.concat([new, existing.iloc[:, append_years]], axis = 1)

    def confirm_match(self, ticker, period):
        if ticker != self.ticker or period != self.period:
            raise ValueError("Ticker and Period must match")

    def selectFolder(self, store):
        return store.financials(self)

    def filename(self):
        return self.ticker + self.period + ".pkl"
 
    def saveTo(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self.to_dict(), file)

    def loadFrom(self, file_path):
        with open(file_path, "rb") as file:
            dictionary = pickle.load(file)
        self.from_dict(dictionary)
        return self

    def to_dict(self):
        return {"ticker" : self.ticker,
                "period" : self.period, 
                "statements" : self.statements}

    def from_dict(self, dictionary):
        ticker = dictionary["ticker"]
        period = dictionary["period"].lower()
        self.confirm_match(ticker, period)
        self.statements = dictionary["statements"]


class PriceHistory(StorageResource):
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = {}

    def selectFolder(self, store):
        return store.priceHistory(self)

    def filename(self):
        return self.ticker + "prices.pkl"

    def loadFrom(self, file_path):
        self.update_prices(pandas.read_pickle(file_path))
        return self

    def saveTo(self, file_path):
        self.data["prices"].to_pickle(file_path)

    def update_prices(self, prices):
        self.data["prices"] = prices

    @property
    def prices(self):
        return self.data["prices"]

    @prices.setter
    def prices(self, new_prices):
        self.data["prices"] = new_prices
           

class WSJinternet():

    def __init__(self):
        self.page_root = "http://quotes.wsj.com/AU/XASX/"
        self.summary_pages = {"overview" : "", 
                              "financials" : "/financials"}
        self.statement_pages = {"income" : "/financials/<period>/income-statement", 
                                "balance" : "/financials/<period>/balance-sheet", 
                                "cashflow" : "/financials/<period>/cash-flow"}
        self.scraper = WSJscraper()
     
        
    def getFinancials(self, ticker, period):
        financials = Financials(ticker, period)

        for sheet in self.statement_pages:
            html = self.load_page(ticker, sheet, period)
            financials.statements[sheet] = self.scraper.getTables(sheet, html)

        return financials

    def get_address(self, ticker, sheet, period = "annual"):
        if period not in ["annual", "quarter", "interim"]:
            raise ValueError("Should be 'annual', 'interim' or 'quarter'")
        if period == "interim":
            period = "quarter"
        address = self.page_root + ticker + self.statement_pages[sheet]
        return  address.replace("<period>", period)

    def load_page(self, ticker, sheet, period):
        try:
            page = requests.get(self.get_address(ticker, sheet, period))
        except requests.HTTPError:
            print("Problem downloading: " + ticker + " " + sheet)
        return page.content



class WSJlocal(WSJinternet):

    def __init__(self):
        self.page_root = "D:\\Investing\\Data\\"
        self.statement_pages = {"income"    :   "\\Financials\\<period>\\<ticker>income.html", 
                                "balance"   :   "\\Financials\\<period>\\<ticker>balance.html",
                                "cashflow"   :   "\\Financials\\<period>\\<ticker>cashflow.html"}
        self.scraper = WSJscraper()

    def load_page(self, ticker, sheet, period):
        location = self.page_root + ticker + self.statement_pages[sheet]
        location = location.replace("<ticker>", ticker)
        location = location.replace("<period>", period)
        try:
            with open(location, 'r') as file:
                page = file.read()
        except:
            print("Problem loading: " + location)
        return page

            

class WSJscraper():

    def __init__(self):
        '''
        statements defines the search terms to look for in each html page to find the table.
        i.e. reading from left to right: in 'income' html, to find 'income' table look for 'Sales/Revenue'.
        '''
        #                   PAGE          TABLE           CONTAINS
        self.statements = {"income"   : {"income"      : "Sales/Revenue"}, 
                           "balance"  : {"assets"      : "Cash & Short Term Investments", 
                                         "liabilities" : "ST Debt & Current Portion LT Debt"}, 
                           "cashflow" : {"operating"   : "Net Operating Cash Flow", 
                                         "investing"   : "Capital Expenditures", 
                                         "financing"   : "Cash Dividends Paid - Total"}}


    def load_overview(self, ticker, store):
        overview = store.html(ticker, "overview")
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

    def getTables(self, sheet, html):
        page = self.statements[sheet]
        scraped_tables = {}
        for table in page:
            search_term = page[table]
            scraped_tables[table] = self.read_statement_table(html, search_term)
        return scraped_tables

    def read_statement_table(self, html, contains):
        try:
            table = pandas.read_html(html, match = contains, index_col = 0)[0]
        except ValueError as E:
            raise MissingStatementEntryError(E.message)
        headings = table.columns.tolist()
        # Delete empty columns after final year
        # First column after final year is trend column
        trend_ix = ["trend" in heading for heading in headings].index(True)
        for heading in headings[trend_ix:]:
            del table[heading]
        # Delete rows starting with 'nan'
        non_nans = [not isinstance(row_label, float) for row_label in table.index]
        table = table.loc[non_nans]
        self.check_years(table.columns.tolist())
        return table

    def check_years(self, years):
        if not all(['20' in year for year in years]):
            raise InsufficientDataError("Empty report years")



class YahooDataDownloader():
    '''
    Uses the Pandas data functionality to download data and handle local storage.
    '''
    def priceHistory(self, ticker, start = None, end = None):
        if start is None:
            start = datetime.date(2010, 01, 01)
        if end is None:
            end = datetime.date.today()
        return pd_data.get_data_yahoo(ticker + ".AX", start, end)

    def currentPrice(self, ticker):
        ticker = ticker + ".AX"
        quote = pd_data.get_quote_yahoo(ticker)
        return quote["last"][ticker]



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

    def load(self, resource):
        folder = resource.selectFolder(self)
        filename = resource.filename()
        return resource.loadFrom(os.path.join(folder, filename))

    def save(self, resource):
        folder = resource.selectFolder(self)
        self.check_directory(folder)
        file_path = os.path.join(folder, resource.filename())
        resource.saveTo(file_path)

    def financials(self, resource):
        return os.path.join(self.root, resource.ticker, "Financials")

    def annualFinancials(self, resource):
        return os.path.join(self.root, resource.ticker, "Financials", "Annual")

    def interimFinancials(self, resource):
        return os.path.join(self.root, resource.ticker, "Financials", "Interim")

    def priceHistory(self, resource):
        return os.path.join(self.root, resource.ticker)

    def check_directory(self, path):
        if "." in os.path.basename(path):
            path = os.path.dirname(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def list_files(self, root_dir, search_term = ""):
        all_files = os.listdir(root_dir)
        return [filename for filename in all_files if search_term in filename]

    def migrate_all(self, folder_pattern, type, tickers = None, file_pattern = None):

        if tickers is None:
            xls = XLSio(self)
            xls.loadWorkbook("StockSummary")
            xls.table = xls.table[xls.table["P/E Ratio (TTM)"].notnull()]
            tickers = xls.getTickers()

        for ticker in tickers:
            folder = folder_pattern.replace("<ticker>", ticker)
            if os.path.exists(folder):
                self.migrate(folder, type, ticker, file_pattern)

    def migrate(self, old_folder, type, ticker, file_pattern = None):

        destination = self.get_folder(ticker, type)

        if file_pattern is not None:
            wanted = lambda name: os.path.isfile(os.path.join(old_folder, name)) and file_pattern in name
            move_files = [file for file in os.listdir(old_folder) if wanted(file)]
            for file in move_files:
                self.migrate_file(old_folder, destination, file)
        else:
            destination_parent = os.path.dirname(destination)
            old_folder_name = os.path.basename(old_folder)
            destination_folder_name = os.path.basename(destination)
            if os.path.dirname(old_folder) != destination_parent:
                self.check_directory(destination)
                shutil.move(old_folder, destination_parent)
            if old_folder_name != destination_folder_name:
                os.rename(os.path.join(destination_parent, old_folder_name), destination)

    def migrate_file(self, old_folder, destination, filename):
        dest_file = os.path.join(destination, filename)
        self.check_directory(dest_file)
        shutil.move(os.path.join(old_folder, filename), dest_file)



class CMCscraper():

    def __init__(self, store):
        self.store = store
        self.root_page = "https://www.cmcmarketsstockbroking.com.au"
        self.login_url = self.root_page + "/login.aspx"
        self.payload = {"logonAccount" : "markhocky", 
                        "logonPassword" : "X", 
                        "source" : "cmcpublic", 
                        "referrer" : self.root_page + "/default.aspx?"}
        self.session = None


    def loginSession(self):
        password = input("Enter password for " + self.payload["logonAccount"])
        self.payload["logonPassword"] = password
        self.session = requests.Session()
        self.session.post(self.login_url, data = self.payload)

    def researchPage(self, ticker):
        research_page = self.root_page + "/net/ui/Research/Research.aspx?asxcode=" + ticker + "&view=historical"
        return research_page

    def download_historicals(self, tickers):
        for ticker in tickers:
            try:
                per_share, historical = self.historicalFigures(ticker)
            except Exception:
                print("No results for " + ticker)
            else:
                per_share_path = self.store.summary_financials(ticker, "pershare")
                historical_path = self.store.summary_financials(ticker, "historical")
                self.store.check_directory(per_share_path)
                self.store.check_directory(historical_path)
                per_share.to_pickle(per_share_path)
                historical.to_pickle(historical_path)

    def historicalFigures(self, ticker):
        if self.session is None:
            self.loginSession()

        page = self.session.get(self.researchPage(ticker))
        soup = BeautifulSoup(page.text, "lxml")
        per_share_stats = pandas.read_html(str(soup.find_all("table")), match = "PER SHARE")[-1]
        per_share_stats = self.cleanTable(per_share_stats)
        historical_financials = pandas.read_html(str(soup.find_all("table")), match = "HISTORICAL")[-1]
        historical_financials = self.cleanTable(historical_financials)

        return (per_share_stats, historical_financials)

    def cleanTable(self, table):
        table_name = table.columns[0]
        dates = table.iloc[1][1:].tolist()
        dates.insert(0, table_name)
        table.columns = dates
        table = table.ix[2:]

        row_labels = table.iloc[:, 0]
        row_labels = [row.replace("\\r\\n", "") for row in row_labels]
        row_labels = [row.replace("\\xa0", " ") for row in row_labels]

        table.index = row_labels
        table = table.iloc[:, 1:]
        table = table.apply(pandas.to_numeric, errors = 'coerce')

        return table


class InsufficientDataError(IOError):
    pass

class MissingStatementEntryError(IOError):
    pass

