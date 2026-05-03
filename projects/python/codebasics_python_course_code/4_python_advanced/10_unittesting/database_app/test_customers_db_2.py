import pytest
from customers_db import CustomersDB

@pytest.fixture
def db():
    db_instance = CustomersDB()
    db_instance.connect()
    yield db_instance
    db_instance.close()


def test_insert_customer(db):
    db.insert_customer("Virat Kohli", "virat@xyz.com")
    customer = db.get_customer_by_name("Virat Kohli")
    assert customer is not None
    assert customer['name'] == "Virat Kohli"
    assert customer['email'] == "virat@xyz.com"
    db.clear_customers()


def test_get_all_customers(db):
    db.insert_customer("Virat Kohli", "virat@xyz.com")
    db.insert_customer("Taylor Swift", "taylor@xyz.com")

    customers = db.get_all_customers()
    assert len(customers) == 2

    db.clear_customers()