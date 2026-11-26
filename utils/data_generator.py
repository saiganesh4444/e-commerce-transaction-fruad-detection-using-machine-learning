"""Synthetic e-commerce transaction generator and model training helpers."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

DEVICES = ["Mobile", "Desktop", "Tablet"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "COD"]
PRODUCT_CATEGORIES = ["Electronics", "Clothing", "Groceries", "Furniture", "Others"]
LOCATIONS = ["Chennai", "Mumbai", "Bangalore", "Delhi", "Hyderabad", "Others"]

FEATURE_NUM_COLS = [
    "Transaction Amount",
    "Quantity",
    "Customer Age",
    "Account Age Days",
    "Transaction Hour",
]

FEATURE_CAT_COLS = [
    "Payment Method",
    "Product Category",
    "Customer Location",
    "Device Used",
]


def generate_synthetic_transactions(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic e-commerce transaction dataset.

    Produces transactions with realistic fraud correlations (high transaction
    amounts, new accounts, late-night hours, high quantities, and high-risk
    device/payment combinations) so the full pipeline works out-of-the-box when
    ``transaction.csv`` is not present locally.
    """
    rng = np.random.default_rng(seed)

    transaction_ids = [f"TXN-{100000 + i}" for i in range(n_rows)]
    amounts = np.round(rng.lognormal(mean=7.6, sigma=1.15, size=n_rows), 2)
    amounts = np.clip(amounts, 50.0, 99500.0)

    quantities = rng.integers(1, 11, size=n_rows)
    ages = rng.integers(18, 76, size=n_rows)
    account_ages = rng.integers(0, 2500, size=n_rows)
    hours = rng.integers(0, 24, size=n_rows)

    devices = rng.choice(DEVICES, size=n_rows, p=[0.56, 0.32, 0.12])
    payments = rng.choice(PAYMENT_METHODS, size=n_rows, p=[0.38, 0.24, 0.26, 0.12])
    categories = rng.choice(PRODUCT_CATEGORIES, size=n_rows, p=[0.28, 0.25, 0.20, 0.15, 0.12])
    locations = rng.choice(LOCATIONS, size=n_rows, p=[0.18, 0.22, 0.22, 0.16, 0.14, 0.08])

    # Risk score logit combining realistic fraud indicators with clean separation
    risk_logit = (
        -6.8
        + 0.00014 * amounts
        + 0.85 * (quantities >= 5).astype(float)
        + 3.20 * (account_ages < 30).astype(float)
        + 2.60 * ((hours >= 0) & (hours <= 4)).astype(float)
        + 0.90 * (categories == "Electronics").astype(float)
        + 0.70 * (devices == "Mobile").astype(float)
        + 0.65 * (payments == "Credit Card").astype(float)
        + 0.55 * (ages < 25).astype(float)
        + rng.normal(0.0, 0.25, size=n_rows)
    )
    prob = 1.0 / (1.0 + np.exp(-risk_logit))
    is_fraud = (prob >= 0.50).astype(int)

    return pd.DataFrame(
        {
            "Transaction ID": transaction_ids,
            "Transaction Amount": amounts,
            "Quantity": quantities,
            "Customer Age": ages,
            "Account Age Days": account_ages,
            "Transaction Hour": hours,
            "Device Used": devices,
            "Payment Method": payments,
            "Product Category": categories,
            "Customer Location": locations,
            "is_fraud": is_fraud,
        }
    )


def load_or_create_dataset(csv_path: str = "transaction.csv", n_rows: int = 8000) -> pd.DataFrame:
    """Load ``csv_path`` if it exists, otherwise return a synthetic dataset."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return generate_synthetic_transactions(n_rows=n_rows)
