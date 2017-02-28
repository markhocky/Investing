from sqlalchemy import Column, ForeignKey, Integer, Numeric, String, Boolean, Date, create_engine
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import date

Base = declarative_base()

class Company(Base):
    __tablename__ = 'company'

    ticker = Column(String(10), primary_key = True)
    name = Column(String(250), nullable = False)
    
class Statement(Base):
    '''
    Statements records are the statements which can be generated for a company.
    Line items appearing on a statement are linked through the StatementItem table.
    '''
    __tablename__ = 'statement'
    
    type = Column(String(50), primary_key = True)
    line_items = relationship("LineItem", secondary = "statement_item")

class LineItem(Base):
    '''
    LineItems specify entries which may exist in a statement.
    For example, Revenue, EBIT.
    The additive field specifies whether the item is additive: True (income)
    or negative: False (expense).
    '''
    __tablename__ = 'line_item'

    id = Column(Integer, primary_key = True)
    name = Column(String(250), nullable = False)
    additive = Column(Boolean)
    companies = relationship("Company", secondary = "statement_fact")


class StatementItem(Base):
    '''
    StatementItems join LineItems and Statements
    '''
    __tablename__ = 'statement_item'
    
    statement_type = Column(Integer, ForeignKey('statement.type'), primary_key = True)
    line_item_id = Column(Integer, ForeignKey('line_item.id'), primary_key = True)
    row_num = Column(Integer, nullable = False)
    statement = relationship(Statement, backref = backref("line_item_assoc"))
    line_item = relationship(LineItem, backref = backref("statement_assoc"))


class StatementFact(Base):
    '''
    StatementFacts record actual results for a given company and time.
    '''
    __tablename__ = 'statement_fact'

    ticker = Column(String(10), ForeignKey('company.ticker'), primary_key = True)
    line_item_id = Column(Integer, ForeignKey('line_item.id'), primary_key = True)
    date = Column(Date, nullable = False)
    value = Column(Numeric)
    company = relationship(Company, backref = backref("line_item_assoc"))
    line_item = relationship(LineItem, backref = backref("company_assoc"))
    
    
def buildTestDB():
        engine = create_engine("sqlite:///")
        Base.metadata.bind = engine
        Base.metadata.create_all()

        db_session = sessionmaker(bind = engine)
        session = db_session()


        session.add(Company(ticker = "MLD", name = "MACA Ltd"))
        session.add(Company(ticker = "CCP", name = "Credit Corp"))

        session.add(Statement(type = "Income"))
        session.add(Statement(type = "Balance"))

        session.add(LineItem(name = "Revenue", additive = True))
        session.add(LineItem(name = "Expenses", additive = False))
        session.add(StatementItem(statement = income, line_item = revenue, row_num = 1))
        session.add(StatementItem(statement = income, line_item = expenses, row_num = 2))

        session.add(LineItem(name = "Assets"))
        session.add(LineItem(name = "Liabilities"))
        session.add(StatementItem(statement = balance, line_item = assets, row_num = 1))
        session.add(StatementItem(statement = balance, line_item = liab, row_num = 2))

        session.add(StatementFact(company = mld, line_item = revenue, date = date(2017, 02, 25), value = 100))
        session.add(StatementFact(company = mld, line_item = expenses, date = date(2017, 02, 25), value = 75))
        session.add(StatementFact(company = mld, line_item = assets, date = date(2017, 02, 25), value = 5000))
        session.add(StatementFact(company = mld, line_item = liab, date = date(2017, 02, 25), value = 2300))

        session.add(StatementFact(company = ccp, line_item = revenue, date = date(2017, 02, 25), value = 80))
        session.add(StatementFact(company = ccp, line_item = expenses, date = date(2017, 02, 25), value = 30))
        session.add(StatementFact(company = ccp, line_item = assets, date = date(2017, 02, 25), value = 5400))
        session.add(StatementFact(company = ccp, line_item = liab, date = date(2017, 02, 25), value = 4300))

        self.session.commit()
        return session


def queryStatement(session, statement_type, ticker):
    result = session.query(StatementItem.row_num, StatementFact.date, LineItem.name, LineItem.additive, StatementFact.value).filter(
        StatementFact.line_item_id == LineItem.id).filter(
        LineItem.id == StatementItem.line_item_id).filter(
        StatementItem.statement.has(Statement.type == statement_type)).filter(
        StatementFact.company.has(Company.ticker == ticker)).all()
    return result

