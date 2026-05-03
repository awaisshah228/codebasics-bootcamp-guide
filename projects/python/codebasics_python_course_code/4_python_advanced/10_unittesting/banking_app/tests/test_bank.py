import pytest
from src.bank import BankAccount


# Test for creating a bank account
def test_create_account():
    account = BankAccount("Mira Patel", 100)
    assert account.owner == "Mira Patel"
    assert account.balance == 100

# Test for depositing money
def test_deposit():
    account = BankAccount("Venkat Nadella")
    account.deposit(50)
    account.deposit(40)
    assert account.balance == 90

    # Test depositing a negative amount (should raise an error)
    with pytest.raises(ValueError):
        account.deposit(-10)

# Test for withdrawing money
def test_withdraw():
    account = BankAccount("Andrea Smith", 100)
    account.withdraw(40)
    assert account.balance == 60

    # Test withdrawing more than the balance (should raise an error)
    with pytest.raises(ValueError):
        account.withdraw(200)

# Test for checking the balance
@pytest.mark.skip(reason="skipping due to xyz reason")
def test_get_balance():
    account = BankAccount("Irfan Pathan", 200)
    assert account.get_balance() == 200
