import pandas as pd

URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"

def top_performers(n: int = 30) -> list:
    """Return the top ``n`` tickers ranked by total return.

    The function downloads the ``all_stocks_5yr.csv`` dataset defined by
    ``URL`` and computes the total return of each ticker using the first and
    last available closing prices. The tickers are then sorted in descending
    order of return and the top ``n`` are returned.
    """

    df = pd.read_csv(URL)

    grouped = df.groupby("Name")
    first_close = grouped.first()["close"]
    last_close = grouped.last()["close"]
    total_return = (last_close - first_close) / first_close

    top = total_return.sort_values(ascending=False).head(n)
    return top.index.tolist()


def main():
    tickers = top_performers(30)
    print("Top 30 tickers:")
    print(tickers)


if __name__ == "__main__":
    main()
