from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class Company(Base):
    __tablename__ = 'company'

    ticker = Column(String(10), primary_key = True)
    name = Column(String(250), nullable = False)
    
class Statement(Base):
    '''
    Statements records are the statements which can be generated for a company.
    '''
    __tablename__ = 'statement'
    
    id = Column(Integer, primary_key = True)
    type = Column(String(50), nullable = False)

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


class StatementItem(Base):
    '''
    StatementItems join LineItems and Statements
    '''
    __tablename__ = 'statement_item'
    
    id = Column(Integer, primary_key = True)
    statement_type = Column(Integer, ForeignKey('statement.id'))
    line_item_id = Column(Integer, ForeignKey('line_item.id'))
    row_order = Column(Integer, nullable = False)
