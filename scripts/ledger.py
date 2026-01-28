import pandas as pd
import os
import dateparser
import re
import requests
import sqlite3
from binance.client import Client
from binance.helpers import date_to_milliseconds
from kucoin.client import Market
from datetime import datetime
from pycoingecko import CoinGeckoAPI


def table_exists(conn, table_name):
    try:
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        result = cursor.fetchone()

        if result is not None:
            return True
        else:
            return False
    except sqlite3.Error as e:
        print("SQLite error:", e)
        return False


def get_days_between_dates(date1: str, date2: str) -> float:
    dif_ms = abs(date_to_milliseconds(date1) - date_to_milliseconds(date2))
    return dif_ms / (1000 * 60 * 60 * 24)


def amount_str_2_float(amount: str) -> float:
    p = re.compile("[0-9,.]+")
    m = p.match(amount)
    float_as_str = m.group().replace(",", "")
    return float(float_as_str)


class PriceFeed:
    FALLBACK_CURRENCY = {'binance': 'USDT',
                         'kucoin': 'USDT',
                         'coingecko': 'EUR'}
    BINANCE_BACKLIST = ['LUNAEUR']

    def __init__(self):
        self.clients = {'binance': Client(),
                        'kucoin': Market(url='https://api.kucoin.com'),
                        'coingecko': CoinGeckoAPI()}
        exchange_info_binance = self.clients['binance'].get_exchange_info()
        exchange_info_kucoin = self.clients['kucoin'].get_symbol_list()
        # coingecko uses ids instead of symbols
        # create dict for translation
        coingecko_coins = self.clients['coingecko'].get_coins_list()
        coingecko_ids = [e['id'] for e in coingecko_coins]
        coingecko_symbols = [e['symbol'] for e in coingecko_coins]
        self.coingecko_dict = dict(zip(coingecko_symbols, coingecko_ids))
        self.symbols = {'binance': [s['symbol'] for s in exchange_info_binance['symbols'] if s['symbol'] not in
                                    self.BINANCE_BACKLIST],
                        'kucoin': [s['symbol'] for s in exchange_info_kucoin],
                        'coingecko': [e.upper() + '-EUR' for e in coingecko_symbols]}
        self.symbols['coingecko'].append('anchorust-EUR')
        self.symbol_delimiter = {'binance': '',
                                 'kucoin': '-',
                                 'coingecko': '-'}
        # coingecko uses ids instead of symbols
        # create dict for translation
        coingecko_coins = self.clients['coingecko'].get_coins_list()
        coingecko_ids = list(reversed([e['id'] for e in coingecko_coins]))
        coingecko_symbols = list(reversed([e['symbol'] for e in coingecko_coins]))
        self.coingecko_dict = dict(zip(coingecko_symbols, coingecko_ids))

    def get_change_factor(self, asset: str, currency: str, date: str = None, die_on_failure: bool = True,
                          force_provider: str = None) -> float:
        for provider in self.clients.keys():
            if not force_provider or force_provider == provider:
                symbol = None
                if currency + self.symbol_delimiter[provider] + asset in self.symbols[provider]:
                    symbol = currency + self.symbol_delimiter[provider] + asset
                elif asset + self.symbol_delimiter[provider] + currency in self.symbols[provider]:
                    symbol = asset + self.symbol_delimiter[provider] + currency
                elif currency != self.FALLBACK_CURRENCY[provider] and asset != self.FALLBACK_CURRENCY[provider]:
                    asset_to_fallback = self.get_change_factor(asset, currency=self.FALLBACK_CURRENCY[provider],
                                                               date=date, die_on_failure=False, force_provider=provider)
                    fallback_to_currency = self.get_change_factor(self.FALLBACK_CURRENCY[provider], currency=currency,
                                                                  date=date, die_on_failure=False,
                                                                  force_provider="binance")
                    if asset_to_fallback != -1 and fallback_to_currency != -1:
                        return asset_to_fallback * fallback_to_currency
                if symbol:
                    if date:
                        symbol_price = self.get_historical_symbol_price(provider, symbol, date)
                    else:
                        symbol_price = self.get_current_symbol_price(provider, symbol)
                    if symbol_price:
                        if symbol.startswith(currency):
                            return 1 / symbol_price
                        else:
                            return symbol_price
                    else:
                        raise ValueError("API call did not return any values.")
        if die_on_failure:
            raise ValueError("No symbol defined for combination of {} and {}".format(asset, currency))
        else:
            return float(-1)

    def get_historical_symbol_price(self, provider: str, symbol: str, date: str) -> float:
        date_ms = date_to_milliseconds(date)
        start_date_ms = date_ms - 60000
        kline = None
        price = None
        if provider == "binance":
            kline = self.clients[provider].get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, start_date_ms,
                                                                 date_ms)
        elif provider == "kucoin":
            kline = self.clients[provider].get_kline(symbol, '1min', startAt=start_date_ms // 1000,
                                                     endAt=date_ms // 1000)
        elif provider == "coingecko":
            price = self.get_symbol_price_from_coingecko(provider, symbol, date=date)

        else:
            raise KeyError("Unknown provider {}!".format(provider))
        if kline:
            return float(kline[0][1])
        elif price:
            return price
        elif "UST" in symbol and "USTC" not in symbol:
            return self.get_historical_symbol_price(provider, symbol.replace("UST", "USTC"), date)
        else:
            raise ValueError(
                "API call to {} did not return any values for symbol {} and date {}.".format(provider, symbol, date))

    def get_current_symbol_price(self, provider: str, symbol: str) -> float:
        symbol_overview = None
        price = None
        if provider == "binance":
            symbol_overview = self.clients[provider].get_symbol_ticker(symbol=symbol)
        elif provider == "kucoin":
            symbol_overview = self.clients[provider].get_ticker(symbol)
        elif provider == "coingecko":
            price = self.get_symbol_price_from_coingecko(provider, symbol)
        else:
            raise KeyError("Unknown provider {}!".format(provider))
        if symbol_overview:
            return float(symbol_overview["price"])
        elif price:
            return price
        else:
            raise ValueError("API call to {} did not return any values for symbol {}.".format(provider, symbol))

    def get_symbol_price_from_coingecko(self, provider: str, symbol: str, date: str = None) -> float:
        asset = symbol.split(self.symbol_delimiter[provider])[0].lower()
        currency = symbol.split(self.symbol_delimiter[provider])[1].lower()
        asset_id = self.coingecko_dict[asset] if asset in self.coingecko_dict.keys() else asset
        if date:
            parsed_date = dateparser.parse(date, settings={'TIMEZONE': "UTC"})
            price = self.clients[provider].get_coin_history_by_id(
                id=asset_id,
                date=parsed_date.date().strftime("%d-%m-%Y"))
            price = float(price['market_data']['current_price'][currency])
        else:
            price = self.clients[provider].get_price(ids=asset_id,
                                                     vs_currencies=currency)
            price = float(price[symbol.split(self.symbol_delimiter[provider])[0].lower()][
                             symbol.split(self.symbol_delimiter[provider])[1].lower()])
        return price


class CryptoLedger:
    CURRENCY = "EUR"
    LEDGER_COLUMNS = ["Date(UTC)", "Asset Amount", "Total (" + CURRENCY + ")"]
    OPEN_POSITION_COLUMNS = ["Date(UTC)", "Asset Amount", CURRENCY + "/Unit"]
    CLOSED_POSITION_COLUMNS = ["Date(UTC) of purchase", "Date(UTC) of sell", "Holding time", "Asset Amount",
                               "Profit/Loss (" + CURRENCY + ")", "Purchase price (" + CURRENCY + ")",
                               "Sell price (" + CURRENCY + ")"]
    DATE_SYNONYMS = ["Time", "Date Updated", "Pair", "Date(UTC)", "time", "Date(UTC+1)", "tradeCreatedAt",
                     "orderCreatedAt"]
    PAIR_SYNONYMS = ["Pair", "Price", "pair", "symbol"]
    ORDER_TYPE_SYNONYMS = ["Side", "Type", "type", "side"]
    TOTAL_AMOUNT_SYNONYMS = ["Filled", "Final Amount", "vol", "size", "Order Amount", "Sell"]
    TOTAL_PRICE_SYNONYMS = ["Total", "Amount", "cost", "funds", "Trading total", "Buy", "averagePrice", "dealFunds"]
    STATUS_SYNONYMS = ["status", "Status"]
    TAX_ALLOWANCE = 600
    DB_SELL_PREFIX = "_SELL_HISTORY"
    DB_BUY_PREFIX = "_BUY_HISTORY"

    def __init__(self, asset: str, price_feed: PriceFeed):
        self.ASSET_NAME = asset
        self.buy_history = pd.DataFrame(columns=self.LEDGER_COLUMNS)
        self.sell_history = pd.DataFrame(columns=self.LEDGER_COLUMNS)
        self.open_positions = pd.DataFrame(columns=self.OPEN_POSITION_COLUMNS)
        self.closed_positions = pd.DataFrame(columns=self.CLOSED_POSITION_COLUMNS)
        self.price_feed = price_feed
        self.db_sell_table = asset + self.DB_SELL_PREFIX
        self.db_buy_table = asset + self.DB_BUY_PREFIX
        self.staking_income = {2021: 0, 2022: 0, 2023: 0, 2024: 0, 2025: 0, 2026: 0, 2027: 0, 2028: 0, 2029: 0}

    def get_asset_name(self) -> str:
        return self.ASSET_NAME

    def calculate_position_from_history(self):
        self.buy_history = self.buy_history.sort_values(by=self.LEDGER_COLUMNS[0], ignore_index=True)
        self.sell_history = self.sell_history.sort_values(by=self.LEDGER_COLUMNS[0], ignore_index=True)
        self.open_positions = pd.DataFrame(columns=self.OPEN_POSITION_COLUMNS)
        self.closed_positions = pd.DataFrame(columns=self.CLOSED_POSITION_COLUMNS)
        for row in self.buy_history.index:
            self.open_positions = self.open_positions.append(pd.DataFrame.from_dict({
                self.OPEN_POSITION_COLUMNS[0]: [self.buy_history[self.LEDGER_COLUMNS[0]][row]],
                self.OPEN_POSITION_COLUMNS[1]: [self.buy_history[self.LEDGER_COLUMNS[1]][row]],
                self.OPEN_POSITION_COLUMNS[2]: [self.buy_history[self.LEDGER_COLUMNS[2]][row] /
                                                self.buy_history[self.LEDGER_COLUMNS[1]][row]]}), ignore_index=True)
        for row in self.sell_history.index:
            sell_amount = self.sell_history[self.LEDGER_COLUMNS[1]][row]
            open_index = self.open_positions.index
            aggregator_index = -1
            asset_amount = 0
            while asset_amount < sell_amount:
                aggregator_index = aggregator_index + 1
                asset_amount = \
                    asset_amount + self.open_positions[self.OPEN_POSITION_COLUMNS[1]][open_index[aggregator_index]]
            remainder = asset_amount - sell_amount
            date_of_purchase = self.open_positions[self.OPEN_POSITION_COLUMNS[0]][open_index[aggregator_index]]
            date_of_sell = self.sell_history[self.LEDGER_COLUMNS[0]][row]
            purchase_price = sum([
                self.open_positions[self.OPEN_POSITION_COLUMNS[1]][open_index[i]] *
                self.open_positions[self.OPEN_POSITION_COLUMNS[2]][open_index[i]]
                if i < aggregator_index else
                (self.open_positions[self.OPEN_POSITION_COLUMNS[1]][open_index[i]] - remainder) *
                self.open_positions[self.OPEN_POSITION_COLUMNS[2]][open_index[i]] for i in range(aggregator_index + 1)])
            sell_price = self.sell_history[self.LEDGER_COLUMNS[2]][row]
            if remainder > 0:
                self.open_positions.loc[open_index[aggregator_index], self.OPEN_POSITION_COLUMNS[1]] = remainder
                self.open_positions = self.open_positions.drop(open_index[:aggregator_index])
            else:
                self.open_positions = self.open_positions.drop(open_index[:aggregator_index + 1])
            self.closed_positions = self.closed_positions.append(pd.DataFrame.from_dict({
                self.CLOSED_POSITION_COLUMNS[0]: [date_of_purchase],
                self.CLOSED_POSITION_COLUMNS[1]: [date_of_sell],
                self.CLOSED_POSITION_COLUMNS[2]: [get_days_between_dates(date_of_purchase, date_of_sell)],
                self.CLOSED_POSITION_COLUMNS[3]: [sell_amount],
                self.CLOSED_POSITION_COLUMNS[4]: [sell_price - purchase_price],
                self.CLOSED_POSITION_COLUMNS[5]: [purchase_price],
                self.CLOSED_POSITION_COLUMNS[6]: [sell_price]}), ignore_index=True)

    def import_from_db(self, conn):
        if table_exists(conn, self.db_buy_table):
            query = f"SELECT * FROM {self.db_buy_table}"
            self.buy_history = pd.read_sql_query(query, conn)
        if table_exists(conn, self.db_sell_table):
            query = f"SELECT * FROM {self.db_sell_table}"
            self.sell_history = pd.read_sql_query(query, conn)

    def export_to_db(self, conn):
        self.buy_history.to_sql(self.db_buy_table, conn, index=False, if_exists='replace')
        self.sell_history.to_sql(self.db_sell_table, conn, index=False, if_exists='replace')

    def import_from_file(self, path: str):
        if path.endswith(".xlsx"):
            df = pd.read_excel(io=path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise KeyError("File format of {} not supported for import.".format(path))
        columns = list(df.columns)
        pair_col_name = [name for name in self.PAIR_SYNONYMS if name in columns][0]
        date_col_name = [name for name in self.DATE_SYNONYMS if name in columns][0]
        date_col_offset = [2 if pair_col_name == date_col_name else 0][0]
        amount_col_name = [name for name in self.TOTAL_AMOUNT_SYNONYMS if name in columns][0]
        price_col_name = [name for name in self.TOTAL_PRICE_SYNONYMS if name in columns][0]
        type_col_name = [name for name in self.ORDER_TYPE_SYNONYMS if name in columns]
        status_col_name = [name for name in self.STATUS_SYNONYMS if name in columns]
        for row in df.index:
            pair = df[pair_col_name][row].replace("/", "").replace("XETHZ", "ETH") \
                .replace("XXBTZ", "BTC").split(" ")[-1]
            if self.ASSET_NAME not in pair or (status_col_name
                                               and (str(df[status_col_name[0]][row]).lower() == "canceled"
                                                    or str(df[status_col_name[0]][row]).lower() == "unknown_status")):
                continue
            change_as = pair.replace('-', '').replace(self.ASSET_NAME, '')
            date_st = df[date_col_name][row + date_col_offset].split(".")[0]
            total_am = amount_str_2_float(str(df[amount_col_name][row])) if pair.startswith(self.ASSET_NAME) else \
                amount_str_2_float(str(df[price_col_name][row]))
            total_pr = amount_str_2_float(str(df[price_col_name][row])) if pair.startswith(self.ASSET_NAME) else \
                amount_str_2_float(str(df[amount_col_name][row]))
            if not type_col_name or (df[type_col_name[0]][row].upper() == "BUY" and pair.startswith(self.ASSET_NAME)) \
                    or (df[type_col_name[0]][row].upper() == "SELL" and pair.endswith(self.ASSET_NAME)) \
                    or (df[type_col_name[0]][row].upper() == "MARKET" and pair.endswith(self.ASSET_NAME)):
                self.buy_history = self.add_order_to_history(self.buy_history, change_as, date_st, total_am, total_pr)
            elif (df[type_col_name[0]][row].upper() == "SELL" and pair.startswith(self.ASSET_NAME)) or \
                    (df[type_col_name[0]][row].upper() == "BUY" and pair.endswith(self.ASSET_NAME)):
                self.sell_history = self.add_order_to_history(self.sell_history, change_as, date_st, total_am, total_pr)
            else:
                raise ValueError("Unknown order type {}".format(df[type_col_name][row]))

    def import_manual_swaps(self, path: str):
        df = pd.read_csv(path)
        sell_col_name = "Sell"
        buy_col_name = "Buy"
        date_col_name = "Date(UTC)"
        amount_buy_col_name = "Amount Buy"
        amount_sell_col_name = "Amount Sell"
        for row in df.index:
            if self.ASSET_NAME not in df[sell_col_name][row] and self.ASSET_NAME not in df[buy_col_name][row]:
                continue
            change_as = df[sell_col_name][row]
            date_st = df[date_col_name][row].split(".")[0]
            total_buy = amount_str_2_float(str(df[amount_buy_col_name][row]))
            total_sell = amount_str_2_float(str(df[amount_sell_col_name][row]))
            if self.ASSET_NAME in df[buy_col_name][row]:
                self.buy_history = self.add_order_to_history(self.buy_history, change_as, date_st, total_buy,
                                                             total_sell)
            elif self.ASSET_NAME in df[sell_col_name][row]:
                self.sell_history = self.add_order_to_history(self.sell_history, change_as, date_st, total_sell,
                                                              total_buy)

    def import_staking_rewards(self, path: str):
        if not path.endswith("csv"):
            return
        df = pd.read_csv(path)
        buy_col_name = "denom"
        date_col_name = "timestamp"
        amount_buy_col_name = "amount"
        for row in df.index:
            asset = "ATOM" if df["from"][row].startswith("cosmos") else "OSMO" if df["from"][row].startswith("osmo") \
                else ""
            if df["type"][row] != "GetReward" or self.ASSET_NAME != asset or \
                    df["denom"][row] != ("u" + self.ASSET_NAME.lower()):
                continue
            change_as = self.CURRENCY
            date_st = df[date_col_name][row].split(".")[0]
            total_buy = amount_str_2_float(str(df[amount_buy_col_name][row]))
            total_sell = 0
            parsed_date = dateparser.parse(date_st, settings={'TIMEZONE': "UTC"})
            delta = datetime.now() - parsed_date
            # coingecko free only returns data from a year ago and less
            if delta.days < 365:
                self.staking_income[parsed_date.year] += \
                    total_buy * self.price_feed.get_change_factor(self.ASSET_NAME, self.CURRENCY, date=date_st,
                                                                  die_on_failure=True, force_provider='coingecko')
            self.buy_history = self.add_order_to_history(self.buy_history, change_as, date_st, total_buy, total_sell)

    def add_order_to_history(self, order_book: pd.DataFrame, change_asset: str, date_str: str, total_amount: float,
                             total_price: float, ):
        if change_asset != self.CURRENCY:
            change_factor = self.price_feed.get_change_factor(change_asset, self.CURRENCY, date=date_str)
        else:
            change_factor = 1
        return order_book.append(pd.DataFrame.from_dict({self.LEDGER_COLUMNS[0]: [date_str],
                                                         self.LEDGER_COLUMNS[1]: [total_amount],
                                                         self.LEDGER_COLUMNS[2]: [total_price * change_factor]}),
                                 ignore_index=True)

    def get_active_amount(self) -> float:
        return self.buy_history["Asset Amount"].sum() - self.sell_history["Asset Amount"].sum()

    def get_current_value(self) -> float:
        return self.get_active_amount() * self.price_feed.get_change_factor(self.ASSET_NAME, self.CURRENCY)

    def get_potential_profit(self) -> float:
        total_amount = self.open_positions[self.OPEN_POSITION_COLUMNS[1]].sum()
        purchase_price = sum([
            self.open_positions[self.OPEN_POSITION_COLUMNS[1]][i] *
            self.open_positions[self.OPEN_POSITION_COLUMNS[2]][i]
            for i in self.open_positions.index])
        average_purchase_price = purchase_price / total_amount
        potential_profit = total_amount * self.price_feed.get_change_factor(self.ASSET_NAME,
                                                                            self.CURRENCY) - purchase_price
        print("Total purchase price:", purchase_price)
        print("Amount:", total_amount)
        print("Average price:", average_purchase_price)
        print("Current price:", self.price_feed.get_change_factor(self.ASSET_NAME, self.CURRENCY))
        print("Potential profit:", potential_profit)
        return potential_profit

    def get_taxable_profit(self, year: int) -> float:
        taxable_events = self.closed_positions.loc[
            self.closed_positions[self.CLOSED_POSITION_COLUMNS[1]].str.startswith(str(year)) &
            (self.closed_positions[self.CLOSED_POSITION_COLUMNS[2]] <= 365)]
        print("TAXABLE PROFIT {}".format(self.ASSET_NAME), taxable_events[self.CLOSED_POSITION_COLUMNS[4]].sum())
        return taxable_events[self.CLOSED_POSITION_COLUMNS[4]].sum()

    def get_tax_free_amount(self, profit_so_far: float) -> float:
        generally_tax_free = 0
        still_tax_free = 0
        new_profit = 0
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        factor = self.price_feed.get_change_factor(self.ASSET_NAME, self.CURRENCY)
        for row in self.open_positions.index:
            if get_days_between_dates(current_time, self.open_positions[self.OPEN_POSITION_COLUMNS[0]][row]) > 365:
                generally_tax_free = generally_tax_free + self.open_positions[self.OPEN_POSITION_COLUMNS[1]][row]
            else:
                purchase_factor = self.open_positions[self.OPEN_POSITION_COLUMNS[2]][row]
                purchase_price = self.open_positions[self.OPEN_POSITION_COLUMNS[1]][row] * purchase_factor

                total_amount = self.open_positions[self.OPEN_POSITION_COLUMNS[1]][row]
                potential_profit = total_amount * factor - purchase_price
                if profit_so_far + new_profit + potential_profit < self.TAX_ALLOWANCE:
                    still_tax_free = still_tax_free + total_amount
                    new_profit = new_profit + potential_profit
                else:
                    still_allowed = self.TAX_ALLOWANCE - profit_so_far - new_profit
                    still_tax_free = still_tax_free + still_allowed / (factor - purchase_factor)
                    new_profit = new_profit + still_allowed
                    break
        if generally_tax_free > 0:
            print("You own {} of {} that may be sold without taxes.".format(generally_tax_free, self.ASSET_NAME))
        if still_tax_free > 0:
            print("With a realized profit of {}, you may still sell {} of {} for {} profit ({} in total).".format(
                profit_so_far, still_tax_free, self.ASSET_NAME, new_profit, still_tax_free * factor))
        return still_tax_free


class LedgerContainer:

    def __init__(self, assets: list, exports: str, db_filename: str, manual_swaps: str, staking_history: str):
        self.asset_names = assets
        self.asset_ledgers = dict()
        filename_table_prefix = '_export_name_table'
        filename_column_name = 'files'
        # query = f"SELECT * FROM {table_name}"
        # dataframe = pd.read_sql_query(query, conn)
        # dataframe.to_sql(table_name, conn, index=False, if_exists='replace')
        price_feed = PriceFeed()
        conn = sqlite3.connect(db_filename)
        for a in self.asset_names:
            ledger = CryptoLedger(a, price_feed)
            filename_table_name = a + filename_table_prefix
            processed_file_list = None
            if table_exists(conn, filename_table_name):
                query = f"SELECT * FROM {filename_table_name}"
                processed_file_list = pd.read_sql_query(query, conn)
                ledger.import_from_db(conn)
            for exp in os.listdir(his):
                if processed_file_list is not None and exp in processed_file_list[filename_column_name].values:
                    continue
                if exp.endswith(".csv") or exp.endswith(".xlsx"):
                    ledger.import_from_file(os.path.join(exports, exp))
            processed_file_list = pd.DataFrame(os.listdir(his), columns=[filename_column_name])
            processed_file_list.to_sql(filename_table_name, conn, index=False, if_exists='replace')
            ledger.export_to_db(conn)
            ledger.import_manual_swaps(manual_swaps)
            for exp in os.listdir(staking_history):
                ledger.import_staking_rewards(os.path.join(staking_history, exp))
            ledger.calculate_position_from_history()
            self.asset_ledgers[a] = ledger
        conn.close()

    def print_summary(self, year: int):
        pd.set_option('display.max_columns', None)
        pot_prof = 0
        real_prof = 0
        for a in self.asset_names:
            print('\n\n\n#### SUMMARY FOR ' + a + ' ####')
            print('\n## BUY OVERVIEW ##')
            print(self.asset_ledgers[a].buy_history.head(30))
            print('\n## SELL OVERVIEW ##')
            print(self.asset_ledgers[a].sell_history.head(30))
            print('\nActive amount: ', self.asset_ledgers[a].get_active_amount())
            print('\n## OPEN POSITIONS ##')
            print(self.asset_ledgers[a].open_positions.head(30))
            print('\n## CLOSED POSITIONS ##')
            print(self.asset_ledgers[a].closed_positions.head(30))
            print('\n## STAKING INCOME ##')
            print(str(self.asset_ledgers[a].staking_income[year]))
            print('\nPOTENTIAL PROFIT ' + a)
            pot_prof = pot_prof + self.asset_ledgers[a].get_potential_profit()
            real_prof = real_prof + self.asset_ledgers[a].get_taxable_profit(year)
        print("\n\n\n")
        print("TOTAL IMAGINARY PROFIT: ", pot_prof)
        print("TOTAL TAXABLE PROFIT: ", real_prof)

    def get_sum_of_taxable_profits(self, year: int) -> float:
        real_prof = 0
        for a in self.asset_names:
            real_prof = real_prof + self.asset_ledgers[a].get_taxable_profit(year)
        return real_prof

    def summarize_sell_options(self, year: int):
        prof = self.get_sum_of_taxable_profits(year)
        for a in self.asset_names:
            self.asset_ledgers[a].get_tax_free_amount(prof)

    def get_total_portfolio_value(self) -> float:
        total_value = 0
        for a in self.asset_names:
            total_value = total_value + self.asset_ledgers[a].get_current_value()
        return total_value

    def get_portfolio_composition(self) -> dict:
        composition = dict()
        total_value = self.get_total_portfolio_value()
        for a in self.asset_names:
            share = self.asset_ledgers[a].get_current_value() / total_value
            composition[a] = share
        return composition


if __name__ == "__main__":
    his = r'/Users/peterpanda/Repos/crypto-ledger/exports'
    his_staking = r'/Users/peterpanda/Repos/crypto-ledger/staking'
    manual_swaps_export = r'/Users/peterpanda/Repos/crypto-ledger/exports/manualSwap/manual.csv'
    asset_list = ['anchorust', 'USDT', 'BUSD', 'ERG', 'ETH', 'BTC', 'HBAR', 'LINK', 'SOL', 'KSM', 'DOT', 'ALGO',
                  'RUNE', 'CKB', 'ADA', 'ATOM', 'OSMO', 'PYTH', 'AVAX', 'NEAR', 'KAS', 'RENDER']
    ledgers = LedgerContainer(asset_list, his, 'ledger_db', manual_swaps_export, his_staking)
    ledgers.print_summary(2025)
    ledgers.summarize_sell_options(2025)
    print('\n## TOTAL VALUE ##')
    print(ledgers.get_total_portfolio_value())
    print('\n## COMPOSITION ## ')
    print(ledgers.get_portfolio_composition())
