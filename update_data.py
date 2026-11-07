
import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta, timezone

# SETTINGS
START_DATE = "2000-01-01"
TICKER = "^IXIC"
VOL_TICKER = "^VXN"
PBR_PROXY_ETF = "QQQ"

# DOWNLOAD
end_date = (
    datetime.now(timezone.utc).date()
    + timedelta(days=1)
).isoformat()

def get_close(ticker):
    df = yf.download(
        ticker,
        start=START_DATE,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"{ticker} data unavailable")

    s = df["Close"]

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    s.index = pd.to_datetime(s.index)

    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)

    s.index = s.index.normalize()
    return s

# NASDAQ + VXN
nasdaq = get_close(TICKER)

data = pd.DataFrame(index=nasdaq.index)
data["NASDAQ"] = nasdaq
data["VXN"] = get_close(VOL_TICKER).reindex(data.index)

# RSI 14
delta = data["NASDAQ"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1/14,
    min_periods=14,
    adjust=False
).mean()

avg_loss = loss.ewm(
    alpha=1/14,
    min_periods=14,
    adjust=False
).mean()

rs = avg_gain / avg_loss
data["RSI_14"] = 100 - (100 / (1 + rs))

# FEAR & GREED
url = (
    "https://raw.githubusercontent.com/"
    "whit3rabbit/fear-greed-data/main/fear-greed.csv"
)

fg = pd.read_csv(url)
fg["Date"] = pd.to_datetime(fg["Date"]).dt.normalize()
fg = fg.set_index("Date").sort_index()
fg = fg.rename(columns={"Fear Greed": "Fear_Greed"})
fg["Fear_Greed"] = pd.to_numeric(
    fg["Fear_Greed"], errors="coerce"
)

data["Fear_Greed"] = (
    fg["Fear_Greed"].reindex(data.index).ffill()
)

# PBR PROXY
pbr = yf.Ticker(PBR_PROXY_ETF).info.get("priceToBook")

if not pbr:
    raise ValueError("QQQ price-to-book unavailable")

prices = data["NASDAQ"].dropna()
anchor_price = float(prices.iloc[-1])

data["PBR"] = pbr * data["NASDAQ"] / anchor_price

# LATEST DATA
final = data.dropna()

if final.empty:
    raise ValueError("No complete data available")

latest = final.iloc[-1]

output = {
    "date": final.index[-1].strftime("%Y-%m-%d"),
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "NASDAQ": round(float(latest["NASDAQ"]), 2),
    "RSI_14": round(float(latest["RSI_14"]), 2),
    "VXN": round(float(latest["VXN"]), 2),
    "Fear_Greed": round(float(latest["Fear_Greed"]), 2),
    "PBR": round(float(latest["PBR"]), 3),
    "PBR_MODE": "proxy"
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print("Dashboard data successfully updated!")
print(json.dumps(output, indent=4))
