import requests
import os
import shutil
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
                            "income" : "/financials/<period>/income-statement", 
                            "balance" : "/financials/<period>/balance-sheet", 
                            "cashflow" : "/financials/<period>/cash-flow"}
        

    def get_address(self, ticker, type, period = "annual"):
        if period not in ["annual", "quarter", "interim"]:
            raise ValueError("Should be 'annual', 'interim' or 'quarter'")
        if period == "interim":
            period = "quarter"
        address = self.page_root + ticker + self.stock_pages[type]
        return  address.replace("<period>", period)

    def download_pages(self, ticker, period):
        types = self.stock_pages.keys()
        if period != "annual":
            types.remove("overview")
            types.remove("financials")
        for type in types:
            try:
                overview = requests.get(self.get_address(ticker, type, period))
                self.save_page(overview.content, self.store.html(ticker, type, period))
            except requests.HTTPError:
                print("Problem downloading: " + ticker + " " + type)
            
    def save_page(self, page, file_path):
        self.store.check_directory(file_path)
        with open(file_path, "w") as file:
            file.write(page)

    def download_all(self, tickers, period):
        for ticker in tickers:
            self.download_pages(ticker, period)


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

    def excel(self, filename = None):
        if filename is None:
            return self.root
        else:
            return os.path.join(self.root, filename + ".xlsx")

    def html(self, ticker, type, period):
        if period == "annual":
            folder = self.get_folder(ticker, "annual financials")
        elif period == "quarter" or period == "interim":
            folder = self.get_folder(ticker, "interim financials")
        return os.path.join(folder, ticker + type + ".html")

    def yahoo(self, ticker):
        folder = self.get_folder(ticker)
        return os.path.join(folder, ticker + "prices.pkl")

    def stock_pickle(self, ticker, type):
        folder = self.get_folder(ticker, "stock_pickle")
        return os.path.join(folder, type + ".pkl")

    def annual_financials(self, ticker, type):
        folder = self.get_folder(ticker, "annual financials")
        return os.path.join(folder, type + ".pkl")

    def interim_financials(self, ticker, type):
        folder = self.get_folder(ticker, "interim financials")
        return os.path.join(folder, type + ".pkl")

    def summary_financials(self, ticker, type):
        folder = self.get_folder(ticker, "summary financials")
        return os.path.join(folder, ticker + type + ".pkl")

    def reports(self, ticker, type, year):
        folder = self.get_folder(ticker, "reports")
        return os.path.join(folder, ticker + type + "report" + year + ".pdf")

    def get_folder(self, ticker, type = None):
        if type == "stock_pickle":
            folder = os.path.join(self.root, ticker, "pickles")
        elif type == "annual financials":
            folder = os.path.join(self.root, ticker, "Financials", "Annual")
        elif type == "interim financials":
            folder = os.path.join(self.root, ticker, "Financials", "Interim")
        elif type == "summary financials":
            folder = os.path.join(self.root, ticker, "Financials")
        elif type == "reports":
            folder = os.path.join(self.root, ticker, "Reports")
        else:
            folder = os.path.join(self.root, ticker)
        return folder

    def check_directory(self, file_path):
        path = os.path.dirname(file_path)
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


class CMCscraper():

    def __init__(self, store):
        self.store = store
        self.root_page = "https://www.cmcmarketsstockbroking.com.au"
        self.login_url = self.root_page + "/login.aspx"
        self.payload = {"logonAccount" : "markhocky", 
                        "logonPassword" : "GreenwaldKahn01", 
                        "source" : "cmcpublic", 
                        "referrer" : self.root_page + "/default.aspx?"}
        self.session = None


    def loginSession(self):
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

