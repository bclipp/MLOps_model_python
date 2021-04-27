from delta.tables import *
import yfinance as yf
import time
import dbutils
import uuid


def main():
    spark = ...
    spark.conf.set("spark.sql.execution.arrow.enabled", True)
    pdf = yf.download(tickers='UBER', period='10y', interval='1d')
    pdf["AdjClose"] = pdf["Adj Close"]
    pdf = pdf.drop("Adj Close", axis=1)
    id = str(uuid.uuid1()).replace('-', '')
    timestamp = int(time.time())
    dbutils.fs.mkdirs("/dbfs/datalake")
    df = spark.createDataFrame(pdf)
    df.write.format("delta").save(f"/dbfs/datalake/stocks_{id}_{timestamp}/data")


if __name__ == "__main__":
    main()
