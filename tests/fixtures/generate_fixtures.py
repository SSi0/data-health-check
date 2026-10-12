"""Generate the fake sample files used by the tests.

Every value is synthetic and deliberately non-real:

* emails use the reserved ``example.com`` / ``example.org`` domains,
* phone numbers use the fictional 555-01xx range,
* SSNs use the 900-999 area numbers, which are never issued as SSNs,
* credit card numbers are the card networks' published test numbers,
* IP addresses come from the documentation ranges (RFC 5737 / RFC 3849).

Usage::

    python tests/fixtures/generate_fixtures.py            # rewrite the fixtures
    python tests/fixtures/generate_fixtures.py --rows 60000 --out big.csv

The second form writes a larger CSV (for memory benchmarks) and is never
committed to the repository.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from openpyxl import Workbook

FIXTURES_DIR = Path(__file__).parent

FIRST_NAMES = [
    "Avery", "Blake", "Carmen", "Dana", "Elliot", "Farah", "Gwen", "Hector",
    "Iris", "Jonah", "Keiko", "Luis", "Mara", "Nikhil", "Olga", "Priya",
    "Quinn", "Rosa", "Samir", "Tess", "Uma", "Victor", "Wen", "Yara",
]  # fmt: skip
LAST_NAMES = [
    "Abbott", "Barnes", "Castillo", "Dunn", "Ellison", "Fischer", "Greene",
    "Hayes", "Ibarra", "Jensen", "Kowalski", "Lindqvist", "Moreau", "Novak",
    "Okafor", "Petrov", "Ramirez", "Sato", "Tran", "Underwood", "Vance",
]  # fmt: skip
STREETS = ["Maple", "Oak", "Cedar", "Birch", "Willow", "Juniper", "Aspen", "Elm"]
STREET_TYPES = ["St", "Ave", "Rd", "Blvd", "Ln", "Way"]
CITIES = [
    ("Springfield", "IL"), ("Riverton", "WY"), ("Fairview", "TX"),
    ("Georgetown", "KY"), ("Salem", "OR"), ("Madison", "WI"), ("Clinton", "NY"),
]  # fmt: skip
AREA_CODES = ["202", "305", "312", "415", "503", "617", "702", "917"]
TEST_CARDS = [
    "4111 1111 1111 1111", "4012 8888 8888 1881", "5555 5555 5555 4444",
    "5105 1051 0510 5100", "3782 822463 10005", "6011 1111 1111 1117",
]  # fmt: skip
PLANS = ["Free", "Starter", "Pro", "Enterprise"]
ORDER_STATUSES = ["shipped", "pending", "delivered", "returned", "cancelled"]
PRODUCTS = ["Widget", "Gadget", "Sprocket", "Gizmo", "Doohickey"]


def _maybe_blank(rng: random.Random, value: object, blank_rate: float) -> object:
    return "" if rng.random() < blank_rate else value


def customer_rows(n: int, seed: int = 7) -> list[dict[str, object]]:
    """Return ``n`` fake customer records."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        city, state = rng.choice(CITIES)
        rows.append(
            {
                "customer_id": 10000 + i,
                "full_name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}{i}@example.com",
                "phone": f"({rng.choice(AREA_CODES)}) 555-01{rng.randint(0, 99):02d}",
                "ssn": f"{rng.randint(900, 999)}-{rng.randint(10, 99)}-"
                f"{rng.randint(1000, 9999)}",
                "street_address": f"{rng.randint(10, 9999)} {rng.choice(STREETS)} "
                f"{rng.choice(STREET_TYPES)}",
                "city": city,
                "state": state,
                "zip_code": f"{rng.randint(10000, 99999)}",
                "credit_card": rng.choice(TEST_CARDS),
                "ip_address": f"192.0.2.{rng.randint(1, 254)}",
                "signup_date": f"2025-{rng.randint(1, 12):02d}-"
                f"{rng.randint(1, 28):02d}",
                "account_balance": _maybe_blank(
                    rng, round(rng.uniform(-200, 5000), 2), 0.08
                ),
                "plan": rng.choice(PLANS),
                "is_active": rng.choice(["yes", "no"]),
                "notes": _maybe_blank(
                    rng, rng.choice(["VIP", "call back", "prefers email"]), 0.7
                ),
            }
        )
    return rows


def order_rows(n: int, n_customers: int, seed: int = 11) -> list[dict[str, object]]:
    """Return ``n`` fake order records referencing fake customer ids."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        quantity = rng.randint(1, 12)
        unit_price = round(rng.uniform(2.5, 250), 2)
        rows.append(
            {
                "order_id": f"ORD-{50000 + i}",
                "customer_id": 10000 + rng.randrange(n_customers),
                "order_date": f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}",
                "product": rng.choice(PRODUCTS),
                "quantity": quantity,
                "unit_price": unit_price,
                "total": round(quantity * unit_price, 2),
                "status": rng.choice(ORDER_STATUSES),
                "contact_email": _maybe_blank(
                    rng, f"buyer{rng.randint(1, 500)}@example.org", 0.15
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(path: Path, sheets: dict[str, list[dict[str, object]]]) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        sheet = workbook.create_sheet(name)
        sheet.append(list(rows[0]))
        for row in rows:
            sheet.append([None if value == "" else value for value in row.values()])
    workbook.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, help="write one large customers CSV")
    parser.add_argument("--out", type=Path, help="output path for --rows")
    args = parser.parse_args()

    if args.rows:
        write_csv(args.out or Path("customers_large.csv"), customer_rows(args.rows))
        return

    write_csv(FIXTURES_DIR / "customers.csv", customer_rows(200))
    write_xlsx(
        FIXTURES_DIR / "company_data.xlsx",
        {"Customers": customer_rows(150, seed=3), "Orders": order_rows(250, 150)},
    )


if __name__ == "__main__":
    main()
