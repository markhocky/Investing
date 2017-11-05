import unittest
import os
from mock import Mock
from DataHandling.Downloads import Storage, Financials
import shutil

class Test_Storage(unittest.TestCase):

    def setUp(self):
        self.root_location = "..\\testData\\"
        self.store = Storage(self.root_location)
        self.resource = Financials("ASX", "annual")

    def test_StorageLoadResource(self):
        self.resource.fromDict = Mock(return_value = self.resource)
        self.store.load_pickle = Mock(return_value = "saved_dict")

        loaded_resource = self.store.load(self.resource)

        self.resource.fromDict.assert_called_once_with("saved_dict")
        self.assertIs(loaded_resource, self.resource)

    def test_StorageLocationForFinancials(self):
        expected_folder = self.root_location + self.resource.ticker + "\\Financials"
        self.assertEqual(self.store.financials(self.resource), expected_folder)

    def test_StorageLocationForAnnualFinancials(self):
        expected_folder = self.root_location + self.resource.ticker + "\\Financials\\Annual"
        self.assertEqual(self.store.annualFinancials(self.resource), expected_folder)

    def test_StorageLocationForInterimFinancials(self):
        expected_folder = self.root_location + self.resource.ticker + "\\Financials\\Interim"
        self.assertEqual(self.store.interimFinancials(self.resource), expected_folder)


if __name__ == '__main__':
    unittest.main()
