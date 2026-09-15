#!/usr/bin/env python3
"""Connect to the local IB Gateway and verify account + market access.

Paper by default. Never places an order unless --demo-order is given AND the
connected account is a paper account (id starts with "DU").
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from ib_async import IB, Contract, Future, Index, LimitOrder, Stock


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002, help="4002=paper, 4001=live")
    p.add_argument("--client-id", type=int, default=11)
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument(
        "--demo-order",
        action="store_true",
        help="Place+cancel a far-off paper limit order (paper accounts only).",
    )
    return p.parse_args()


def front_month_future(ib: IB, symbol: str, exchange: str) -> Contract | None:
    details = ib.reqContractDetails(
        Future(symbol=symbol, exchange=exchange, currency="USD")
    )
    today = dt.date.today().strftime("%Y%m%d")
    contracts = [
        d.contract
        for d in details
        if d.contract.lastTradeDateOrContractMonth >= today
    ]
    contracts.sort(key=lambda c: c.lastTradeDateOrContractMonth)
    return contracts[0] if contracts else None


def snapshot(ib: IB, contract: Contract) -> str:
    ticker = ib.reqMktData(contract, snapshot=True)
    ib.sleep(3)
    price = ticker.marketPrice()
    if price != price or price <= 0:  # NaN or empty
        price = ticker.close if ticker.close == ticker.close else None
    ib.cancelMktData(contract)
    return f"{price}" if price else "n/a"


def main() -> int:
    args = parse_args()
    ib = IB()

    print(f"Connecting to {args.host}:{args.port} (clientId={args.client_id}) ...")
    try:
        ib.connect(
            args.host, args.port, clientId=args.client_id, timeout=args.timeout
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to connect: {exc}")
        print("Is the gateway up and logged in? Check: docker logs ibkr-gateway")
        return 1

    print(f"Connected. Server version: {ib.client.serverVersion()}")
    accounts = ib.managedAccounts()
    print(f"Accounts: {accounts}")

    for account in accounts:
        values = {v.tag: v.value for v in ib.accountSummary(account)}
        net_liq = values.get("NetLiquidation", "n/a")
        currency = values.get("Currency", "")
        print(f"  {account}: NetLiquidation={net_liq} {currency}")

    positions = ib.positions()
    print(f"Open positions: {len(positions)}")
    for pos in positions[:10]:
        print(
            f"  {pos.contract.localSymbol or pos.contract.symbol}: "
            f"{pos.position} @ {pos.avgCost}"
        )

    # Delayed data avoids requiring paid market-data subscriptions for a smoke test.
    ib.reqMarketDataType(3)
    print("\nQualifying contracts (delayed data):")

    checks: list[tuple[str, Contract]] = [
        ("Stock AAPL", Stock("AAPL", "SMART", "USD")),
        ("Index SPX", Index("SPX", "CBOE", "USD")),
    ]

    es = front_month_future(ib, "ES", "CME")
    if es:
        checks.append(("Future ES (front month)", es))
    gc = front_month_future(ib, "GC", "COMEX")
    if gc:
        checks.append(("Future GC (front month)", gc))

    for label, contract in checks:
        try:
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                print(f"  {label}: could not qualify")
                continue
            con = qualified[0]
            print(
                f"  {label}: conId={con.conId} "
                f"{con.localSymbol or con.symbol} last={snapshot(ib, con)}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {label}: error {exc}")

    if args.demo_order:
        if not accounts or not accounts[0].startswith("DU"):
            print("\nRefusing demo order: not a paper account (id must start with DU).")
        else:
            contract = ib.qualifyContracts(Stock("AAPL", "SMART", "USD"))[0]
            ticker = ib.reqMktData(contract, snapshot=True)
            ib.sleep(3)
            ref = ticker.marketPrice()
            ib.cancelMktData(contract)
            if ref != ref or ref <= 0:
                print("\nCannot place demo order: no reference price.")
            else:
                limit = round(ref * 0.5, 2)
                order = LimitOrder("BUY", 1, limit)
                trade = ib.placeOrder(contract, order)
                ib.sleep(2)
                print(f"\nDemo order: BUY 1 AAPL @ {limit} -> status={trade.orderStatus.status}")
                ib.cancelOrder(order)
                ib.sleep(2)
                print(f"Demo order cancelled -> status={trade.orderStatus.status}")

    ib.disconnect()
    print("\nOK - gateway reachable, account and contracts working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
