# src/customers_db.py

class CustomersDB:
    def __init__(self):
        """Initialize an in-memory list to simulate a database."""
        self.customers = []
        self.next_id = 1
        self.connection = None

    def connect(self):
        """Simulate connecting to a database."""
        self.connection = "DummyConnectionObject"
        print("Connected to the database.")

    def insert_customer(self, name, email):
        """Insert a new customer into the list."""
        customer = {
            "id": self.next_id,
            "name": name,
            "email": email
        }
        self.customers.append(customer)
        self.next_id += 1

    def get_all_customers(self):
        """Retrieve all customers from the list."""
        return self.customers

    def get_customer_by_name(self, name):
        """Retrieve a customer by name."""
        for customer in self.customers:
            if customer["name"] == name:
                return customer
        return None

    def clear_customers(self):
        """Clear all customers (reset the database)."""
        self.customers = []
        self.next_id = 1

    def close(self):
        """Simulate closing the database connection."""
        self.connection = None
        print("Database connection closed.")