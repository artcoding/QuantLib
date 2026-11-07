"""Print the Ayache–Forsyth–Vetzal Table 2 total-default price at S = 100.

The targets are the hedge-model total-default column. A 200-node grid should
land within 0.05 of 122.7341, and an 800-node grid within 0.02 of 122.7325.
"""

from total_default import TABLE2_TOTAL_DEFAULT, price_at_table_spot


def main() -> None:
    print(f"{'nodes':>8} {'V(100)':>12} {'Table 2':>12} {'diff':>10}")
    for nodes in (200, 400, 800):
        price = price_at_table_spot(nodes)
        target = TABLE2_TOTAL_DEFAULT[nodes]
        print(f"{nodes:8d} {price:12.4f} {target:12.4f} {price - target:+10.4f}")


if __name__ == "__main__":
    main()
