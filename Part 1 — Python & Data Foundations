# Task 1.1 — Read the file the foundations way

import csv
import json

def parse_row(row):
    """Convert numeric fields to int/float without mutating the original row."""
    parsed = dict(row)
    parsed["age"] = int(parsed["age"])
    parsed["fee_inr"] = int(parsed["fee_inr"])
    parsed["prior_no_shows"] = int(parsed["prior_no_shows"])
    parsed["distance_km"] = float(parsed["distance_km"])
    return parsed

accepted = []
rejected = []

with open("/appointments.csv", mode="r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    rows = list(reader)

print("Header:", header)
print("Total rows read:", len(rows))

for row in rows:
    try:
        accepted.append(parse_row(row))
    except ValueError:
        rejected.append(row)

print(f"Accepted: {len(accepted)} | Rejected: {len(rejected)}")

with open("rejected_rows.json", mode="w", encoding="utf-8") as f:
    json.dump(rejected, f, indent=2)

print("Saved rejected_rows.json")


# Inspect and clean, pass 1

import pandas as pd
import numpy as np

# Start fresh from the raw CSV
df = pd.read_csv("/appointments.csv")

print("Raw shape:", df.shape)
print("\n--- DataFrame Info ---")
df.info()
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\nExact duplicates:", df.duplicated().sum())
print("Raw booking_channel unique count:", df["booking_channel"].nunique())

# (a) Drop exact duplicate rows
df = df.drop_duplicates().copy()
print(f"\nAfter (a) dropping duplicates: {len(df)} rows")

# (b) Clean booking_channel
df["booking_channel"] = df["booking_channel"].str.strip().str.title()
print("Cleaned booking_channel unique values:", sorted(df["booking_channel"].unique()))
print(f"After (b) cleaning booking_channel: {len(df)} rows")

# (c) Convert date columns
df["appt_date"] = pd.to_datetime(df["appt_date"], errors="coerce")
df["booked_on"] = pd.to_datetime(df["booked_on"])
print(f"After (c) date conversion (unreadable appt_date = {df['appt_date'].isna().sum()}): {len(df)} rows")

# (d) Drop rows where appt_date or status is missing
df = df.dropna(subset=["appt_date", "status"]).copy()
print(f"After (d) dropping missing appt_date/status: {len(df)} rows")

# Task 1.3 — Clean, pass 2: validity, gaps and outliers

# (a) Mark invalid ages (< 1 or > 100) as missing
invalid_age_mask = (df["age"] < 1) | (df["age"] > 100)
print("Invalid ages found:", invalid_age_mask.sum())
df.loc[invalid_age_mask, "age"] = np.nan
print("Total missing ages before fill:", df["age"].isna().sum())

# (b) Fill missing age and distance_km with their medians
median_age = df["age"].median()
median_dist = df["distance_km"].median()
print(f"Median age: {median_age} | Median distance_km: {median_dist}")

df["age"] = df["age"].fillna(median_age).round().astype(int)
df["distance_km"] = df["distance_km"].fillna(median_dist)

# (c) Compute Q1, Q3, IQR and cap outliers with np.clip
for col in ["distance_km", "fee_inr"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    capped_count = ((df[col] < lower_fence) | (df[col] > upper_fence)).sum()
    df[col] = np.clip(df[col], lower_fence, upper_fence)

    print(
        f"{col}: Q1={q1}, Q3={q3}, IQR={iqr}, "
        f"lower={lower_fence}, upper={upper_fence}, capped={capped_count}"
    )

# (d) Round distance_km to 1 decimal and fee_inr to whole rupees
df["distance_km"] = df["distance_km"].round(1)
df["fee_inr"] = df["fee_inr"].round().astype(int)

print("Max fee_inr after capping & rounding:", df["fee_inr"].max())

# Task 1.4 — Features and a NumPy summary

# Create derived features
df["lead_days"] = (df["appt_date"] - df["booked_on"]).dt.days
df["no_show"] = (df["status"] == "No-Show").astype(int)

print(f"lead_days range: {df['lead_days'].min()} to {df['lead_days'].max()}")
print(f"No-shows: {df['no_show'].sum()} / {len(df)} ({df['no_show'].mean() * 100:.1f}%)")

# Vectorized NumPy summary (no loops)
X = df[["age", "distance_km", "fee_inr", "lead_days"]].to_numpy()

# (a) Column means
col_means = np.mean(X, axis=0)
print("Column means [age, distance_km, fee_inr, lead_days]:", np.round(col_means, 2))

# (b) Rows with lead_days > 14
lead_mask = X[:, 3] > 14
print(f"lead_days > 14: {lead_mask.sum()} rows ({lead_mask.mean() * 100:.1f}%)")

# (c) Z-scores for distance_km using population std (np.std default)
dist_col = X[:, 1]
z_scores = (dist_col - np.mean(dist_col)) / np.std(dist_col)
outlier_z_count = (np.abs(z_scores) > 2).sum()
print(f"Rows with |z| > 2 for distance_km: {outlier_z_count}")

# Task 1.5 — Enrich the clinics from a live API

import requests

clinics = pd.read_csv("/clinics.csv")
temps = []

url = "https://api.open-meteo.com/v1/forecast"

for _, row in clinics.iterrows():
    params = {
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "hourly": "temperature_2m",
        "forecast_days": 1,
    }
    temp_val = None
    last_error = None

    for attempt in range(2):  # Initial try + 1 retry
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                payload = resp.json()
                if "hourly" in payload and "temperature_2m" in payload["hourly"]:
                    temp_val = payload["hourly"]["temperature_2m"][0]
                    break
                else:
                    last_error = "Missing 'hourly' or 'temperature_2m' in response"
            else:
                last_error = f"HTTP {resp.status_code}"
        except requests.exceptions.RequestException as e:
            last_error = str(e)

    if temp_val is None:
        print(f"Failed for {row['clinic_name']}: {last_error}")
    else:
        print(f"{row['clinic_name']} ({row['city']}): {temp_val} °C")

    temps.append(temp_val)

clinics["temp_c"] = temps
clinics.to_csv("clinics_enriched.csv", index=False)
print("\nSaved clinics_enriched.csv")
print(clinics)

# Task 1.6 — Export the clean table and write the SQL load script

# Sort by appointment_id and format dates as YYYY-MM-DD
df_clean = df.sort_values("appointment_id").copy()
df_clean["booked_on"] = df_clean["booked_on"].dt.strftime("%Y-%m-%d")
df_clean["appt_date"] = df_clean["appt_date"].dt.strftime("%Y-%m-%d")

# Enforce exact 16-column order
cols_16 = [
    "appointment_id", "patient_id", "clinic_id", "booked_on", "appt_date",
    "age", "gender", "distance_km", "fee_inr", "booking_channel",
    "visit_type", "reminder_sent", "prior_no_shows", "status",
    "lead_days", "no_show"
]
df_clean = df_clean[cols_16]

# Pre-export integrity checks
print("Distinct appointment_id:", df_clean["appointment_id"].nunique())
print("Total missing values across 16 columns:", df_clean.isna().sum().sum())
assert df_clean["appointment_id"].is_unique, "appointment_id is not unique!"
assert df_clean.isna().sum().sum() == 0, "Missing values remain!"

# Save clean CSV
df_clean.to_csv("appointments_clean.csv", index=False)
print(f"Saved appointments_clean.csv with shape {df_clean.shape}")

# Generate load_data.sql (clinics first, then appointments)
sql_lines = []

for _, r in clinics.iterrows():
    temp_sql = "NULL" if pd.isna(r["temp_c"]) else f"{r['temp_c']}"
    stmt = (
        f"INSERT INTO clinics VALUES ("
        f"'{r['clinic_id']}','{r['clinic_name']}','{r['city']}','{r['tier']}',"
        f"{r['latitude']},{r['longitude']},{temp_sql});"
    )
    sql_lines.append(stmt)

for _, r in df_clean.iterrows():
    stmt = (
        f"INSERT INTO appointments VALUES ("
        f"'{r['appointment_id']}','{r['patient_id']}','{r['clinic_id']}',"
        f"'{r['booked_on']}','{r['appt_date']}',{int(r['age'])},'{r['gender']}',"
        f"{r['distance_km']},{int(r['fee_inr'])},'{r['booking_channel']}',"
        f"'{r['visit_type']}','{r['reminder_sent']}',{int(r['prior_no_shows'])},"
        f"'{r['status']}',{int(r['lead_days'])},{int(r['no_show'])});"
    )
    sql_lines.append(stmt)

with open("load_data.sql", mode="w", encoding="utf-8") as f:
    f.write("\n".join(sql_lines) + "\n")

print(f"Saved load_data.sql with {len(sql_lines)} INSERT statements (expected 998)")
