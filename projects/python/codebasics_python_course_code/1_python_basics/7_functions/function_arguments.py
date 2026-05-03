def sum_all(*args):
    total = 0
    for num in args:
        total += num
    return total

total = sum_all(1,2,3,4,5)

print(f"Sum of all numbers: {total}")

def company_info(**kwargs):
    if 'ticker' in kwargs:
        print("Ticker: ", kwargs['ticker'])
    if 'ceo' in kwargs:
        print("CEO: ", kwargs['ceo'])
    if 'revenue' in kwargs:
        print("Revenue:", kwargs['revenue'])


company_info(ticker='AAPL', ceo="Tim Cook", revenue="200 billion")
    