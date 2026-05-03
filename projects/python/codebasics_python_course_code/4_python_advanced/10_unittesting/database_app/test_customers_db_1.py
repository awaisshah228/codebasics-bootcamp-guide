import pytest
from customers_db import CustomersDB


def test_insert_customer():
    db = CustomersDB()
    db.connect()

    db.insert_customer("Virat Kohli", "virat@xyz.com")
    customer = db.get_customer_by_name("Virat Kohli")
    assert customer is not None
    assert customer['name'] == "Virat Kohli"
    assert customer['email'] == "virat@xyz.com"

    db.close()

def test_get_all_customers():
    db = CustomersDB()
    db.connect()

    db.insert_customer("Virat Kohli", "virat@xyz.com")
    db.insert_customer("Taylor Swift", "taylor@xyz.com")

    customers = db.get_all_customers()
    assert len(customers) == 2

    db.close()