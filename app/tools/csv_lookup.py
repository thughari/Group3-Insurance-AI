import os
import pandas as pd
from typing import Dict, List

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RISK_CSV_PATH = os.path.join(DATA_DIR, "RiskScore_Classification_Table.csv")
PREMIUM_CSV_PATH = os.path.join(DATA_DIR, "PremiumRate_ReferenceTable.csv")

def classify_risk(disclosures: List[str]) -> str:
    if not disclosures:
        return "standard"

    try:
        df = pd.read_csv(RISK_CSV_PATH)
    except FileNotFoundError:
        return "unknown"

    tier_scores = {"standard": 0, "substandard": 1, "high": 2, "declined": 3}
    highest_score = 0
    highest_tier = "standard"

    # Normalize disclosures for matching
    lower_disclosures = [d.lower() for d in disclosures]

    for d in lower_disclosures:
        if "smoker" in d and "non-smoker" not in d and "non smoker" not in d:
            # Direct rule for smoker if not in CSV match easily
            score = tier_scores["substandard"]
            if score > highest_score:
                highest_score = score
                highest_tier = "substandard"

        # Try to find a match in the specific_condition column
        matches = df[df["specific_condition"].str.lower().str.contains(d, na=False)]
        if not matches.empty:
            for _, row in matches.iterrows():
                csv_tier = str(row["risk_tier"]).lower()
                
                # Map CSV tier to standard model tier
                if "decline" in csv_tier or "postpone" in csv_tier:
                    mapped_tier = "declined"
                elif "class iii" in csv_tier or "class iv" in csv_tier:
                    mapped_tier = "high"
                elif "class i" in csv_tier or "class ii" in csv_tier or "smoker" in csv_tier:
                    mapped_tier = "substandard"
                else:
                    mapped_tier = "standard"
                
                score = tier_scores[mapped_tier]
                if score > highest_score:
                    highest_score = score
                    highest_tier = mapped_tier

    return highest_tier

def indicative_premium_lookup(age: int, cover_amount: int, term_years: int, risk_tier: str) -> Dict[str, str]:
    try:
        df = pd.read_csv(PREMIUM_CSV_PATH)
    except FileNotFoundError:
        return {"monthly_estimate": "N/A", "disclaimer": "Data unavailable."}

    age = int(age)
    cover_amount = int(cover_amount)
    term_years = int(term_years)

    # Find closest matching age, term, cover safely
    available_ages = [int(x) for x in df["entry_age"].unique() if str(x).replace('.','',1).isdigit()]
    closest_age = min(available_ages, key=lambda x: abs(x - age)) if available_ages else age

    available_terms = [int(x) for x in df["policy_term_years"].unique() if str(x).replace('.','',1).isdigit()]
    closest_term = min(available_terms, key=lambda x: abs(x - term_years)) if available_terms else term_years

    available_covers = [int(x) for x in df["sum_assured_inr"].unique() if str(x).replace('.','',1).isdigit()]
    closest_cover = min(available_covers, key=lambda x: abs(x - cover_amount)) if available_covers else cover_amount

    # Filter
    filtered = df[
        (df["entry_age"] == closest_age) & 
        (df["policy_term_years"] == closest_term) & 
        (df["sum_assured_inr"] == closest_cover) &
        (df["gender"] == "Male") # default to male if unknown
    ]

    # Map risk_tier to loading_class (simplistic mapping)
    if risk_tier == "high":
        # Usually high risk might not be in the standard table, but let's take the max available
        row = filtered.sort_values(by="monthly_premium_inr", ascending=False).head(1)
    elif risk_tier == "substandard":
        # Take a row with Class I or Smoker
        row = filtered[filtered["loading_class"].str.contains("Class I", na=False)]
        if row.empty:
            row = filtered.sort_values(by="monthly_premium_inr", ascending=False).head(1)
    else:
        # standard
        row = filtered[filtered["loading_class"] == "Standard"]
        if row.empty:
            row = filtered.sort_values(by="monthly_premium_inr", ascending=True).head(1)

    if not row.empty:
        monthly = row.iloc[0]["monthly_premium_inr"]
        return {
            "monthly_estimate": f"INR {monthly}",
            "disclaimer": "Indicative estimate only. Final premium is subject to underwriting review."
        }

    # Fallback calculation if no rows match
    base = (cover_amount / 100000) * (age * 0.8 + term_years * 1.2)
    multiplier = {"standard": 1.0, "substandard": 1.35, "high": 1.75, "declined": 0.0}.get(risk_tier, 1.0)
    monthly = round(base * multiplier, 2)
    return {
        "monthly_estimate": f"INR {monthly:.2f}",
        "disclaimer": "Indicative calculated estimate only. Final premium is subject to underwriting review."
    }
