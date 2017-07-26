import unittest
import DataHandling.DbWrapper as db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import date

class Test_DbWrapper(unittest.TestCase):
    
    def setUp(self):
        engine = create_engine("sqlite:///")
        Base = declarative_base()
        Base.metadata.bind = engine
        Base.metadata.create_all()

        db_session = sessionmaker(bind = engine)
        self.session = db_session()


        self.session.add(db.Company(ticker = "MLD", name = "MACA Ltd"))
        self.session.add(db.Company(ticker = "CCP", name = "Credit Corp"))

        self.session.add(db.Statement(type = "Income"))
        self.session.add(db.Statement(type = "Balance"))

        self.session.add(db.LineItem(name = "Revenue", additive = True))
        self.session.add(db.LineItem(name = "Expenses", additive = False))
        self.session.add(db.StatementItem(statement = income, line_item = revenue, row_num = 1))
        self.session.add(db.StatementItem(statement = income, line_item = expenses, row_num = 2))

        self.session.add(db.LineItem(name = "Assets"))
        self.session.add(db.LineItem(name = "Liabilities"))
        self.session.add(db.StatementItem(statement = balance, line_item = assets, row_num = 1))
        self.session.add(db.StatementItem(statement = balance, line_item = liab, row_num = 2))

        self.session.add(db.StatementFact(company = mld, line_item = revenue, date = date(2017, 02, 25), value = 100))
        self.session.add(db.StatementFact(company = mld, line_item = expenses, date = date(2017, 02, 25), value = 75))
        self.session.add(db.StatementFact(company = mld, line_item = assets, date = date(2017, 02, 25), value = 5000))
        self.session.add(db.StatementFact(company = mld, line_item = liab, date = date(2017, 02, 25), value = 2300))

        self.session.add(db.StatementFact(company = ccp, line_item = revenue, date = date(2017, 02, 25), value = 80))
        self.session.add(db.StatementFact(company = ccp, line_item = expenses, date = date(2017, 02, 25), value = 30))
        self.session.add(db.StatementFact(company = ccp, line_item = assets, date = date(2017, 02, 25), value = 5400))
        self.session.add(db.StatementFact(company = ccp, line_item = liab, date = date(2017, 02, 25), value = 4300))

        self.session.commit()


    def test_QueryStatement(self):
        self.fail("Not implemented")



if __name__ == '__main__':
    unittest.main()
