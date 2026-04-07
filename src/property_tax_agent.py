#!/usr/bin/env python3
"""
Property Tax Agent — estimates property tax for commercial real estate.
Usage:
    python property_tax_agent.py
    python property_tax_agent.py --address "1283 Industrial Ave, Austin, TX" \
        --sqft 5570 --price-per-sqft 47 --noi 16394 --cap-rate 6.26
"""

import argparse
import anthropic


SYSTEM_PROMPT = """You are a commercial real estate property tax agent specializing in the United States.

When given property details, you will:
1. Calculate the estimated market value (NOI / cap rate, or sqft × price/sqft — use whichever is available)
2. Look up or estimate the local combined property tax rate for the city and county (City + County + School District + special districts)
3. Apply that rate to the assessed value (assume 100% of market value for commercial unless you know otherwise)
4. Return a clean, structured breakdown

Always include:
- Estimated market value
- Assessed value
- Combined effective tax rate (with rate components listed)
- Annual property tax estimate
- Monthly equivalent
- Tax per sqft
- Tax as % of NOI (if NOI is provided)
- A brief note on assumptions or caveats (appraisal district, exemptions, rate year)

Be specific about which jurisdiction's rates you are using and the year. If you are uncertain about the exact current rate, give your best estimate and note the uncertainty."""


def run_agent(address: str, sqft: float | None, price_per_sqft: float | None,
              noi: float | None, cap_rate: float | None) -> None:

    parts = [f"Property address: {address}"]
    if sqft:
        parts.append(f"Building size: {sqft:,.0f} sqft")
    if price_per_sqft:
        parts.append(f"Price per sqft: ${price_per_sqft}/sqft")
    if noi:
        parts.append(f"Annual NOI: ${noi:,.0f}")
    if cap_rate:
        parts.append(f"Cap rate: {cap_rate}%")

    user_message = "Please calculate the estimated property tax for this commercial property:\n\n" + "\n".join(parts)

    client = anthropic.Anthropic()

    print("\n" + "─" * 60)
    print("  Commercial Property Tax Agent")
    print("─" * 60)
    print(f"  {address}")
    print("─" * 60 + "\n")

    with client.messages.stream(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print("\n\n" + "─" * 60 + "\n")


def interactive_mode() -> None:
    print("\n" + "═" * 60)
    print("  Commercial Property Tax Agent — Interactive Mode")
    print("═" * 60)
    print("  Enter property details (press Enter to skip optional fields)\n")

    address = input("  Address (required): ").strip()
    if not address:
        print("  Address is required. Exiting.")
        return

    def prompt_float(label: str) -> float | None:
        val = input(f"  {label}: ").strip()
        try:
            return float(val.replace(",", "").replace("$", "").replace("%", "")) if val else None
        except ValueError:
            return None

    sqft           = prompt_float("Building size (sqft)")
    price_per_sqft = prompt_float("Price per sqft ($)")
    noi            = prompt_float("Annual NOI ($)")
    cap_rate       = prompt_float("Cap rate (%)")

    run_agent(address, sqft, price_per_sqft, noi, cap_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate property tax for a commercial property.")
    parser.add_argument("--address",        type=str,   help="Full property address")
    parser.add_argument("--sqft",           type=float, help="Building size in sqft")
    parser.add_argument("--price-per-sqft", type=float, help="Asking price per sqft")
    parser.add_argument("--noi",            type=float, help="Annual NOI in dollars")
    parser.add_argument("--cap-rate",       type=float, help="Cap rate as a percentage (e.g. 6.26)")
    args = parser.parse_args()

    if args.address:
        run_agent(args.address, args.sqft, args.price_per_sqft, args.noi, args.cap_rate)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
