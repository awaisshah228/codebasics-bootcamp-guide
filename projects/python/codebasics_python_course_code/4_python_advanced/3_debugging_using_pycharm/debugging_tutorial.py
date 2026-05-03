def calculate_total(prices, discount):
    total = 0
    for price in prices:
        total += price
    total -= discount
    return total


if __name__ == "__main__":
    prices = [10.99, 5.49, 3.99, 2.49]
    discount = 2.00
    total_price = calculate_total(prices, discount)
    print(f"Total price after discount: ${total_price:.2f}")