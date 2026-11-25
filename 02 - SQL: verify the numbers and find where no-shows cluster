# Part 2 — SQL: verify the numbers and find where no-shows cluster

# CareLoop Clinics  No Show Guard | Part 2 SQL Verification

# Task 2_1  Build and load the database
# (Skip the next two lines if running in SQLiteOnline)

import sqlite3
import pandas as pd

# Create a fresh SQLite database in Colab
conn = sqlite3.connect(":memory:")
cur = conn.cursor()

# Enforce foreign keys in SQLite
cur.execute("PRAGMA foreign_keys = ON;")

# Create tables
cur.executescript("""
CREATE TABLE clinics (
    clinic_id   VARCHAR(5) PRIMARY KEY,
    clinic_name VARCHAR(40) NOT NULL,
    city        VARCHAR(20) NOT NULL,
    tier        VARCHAR(10),
    latitude    DECIMAL(8,4),
    longitude   DECIMAL(8,4),
    temp_c      DECIMAL(5,1)
);

CREATE TABLE appointments (
    appointment_id  VARCHAR(8) PRIMARY KEY,
    patient_id      VARCHAR(8) NOT NULL,
    clinic_id       VARCHAR(5) NOT NULL,
    booked_on       DATE,
    appt_date       DATE,
    age             INT,
    gender          VARCHAR(1),
    distance_km     DECIMAL(5,1),
    fee_inr         INT,
    booking_channel VARCHAR(10),
    visit_type      VARCHAR(12),
    reminder_sent   VARCHAR(3),
    prior_no_shows  INT,
    status          VARCHAR(10),
    lead_days       INT,
    no_show         INT,
    FOREIGN KEY (clinic_id) REFERENCES clinics(clinic_id)
);
""")

# Load data from load_data.sql (uploaded/generated in Colab)
with open("load_data.sql", "r", encoding="utf-8") as f:
    cur.executescript(f.read())

# Verify counts (Expected: 8 clinics, 990 appointments, 218 no-shows)
print("Clinics:", pd.read_sql_query("SELECT COUNT(*) AS clinic_count FROM clinics;", conn).to_string(index=False))
print("\nAppointments & No-Shows:")
print(pd.read_sql_query("SELECT COUNT(*) AS appointments, SUM(no_show) AS no_shows FROM appointments;", conn).to_string(index=False))

# Test duplicate primary key ('A0001')
try:
    cur.execute("""
        INSERT INTO appointments VALUES (
            'A0001', 'P999', 'C01', '2025-01-01', '2025-01-05',
            35, 'M', 5.0, 500, 'App', 'Consultation', 'Yes', 0, 'Attended', 4, 0
        );
    """)
except sqlite3.IntegrityError as e:
    print("\nDuplicate PK Error Caught:", e);

# Task 2_2  No show rate by clinic

q_2_2a = """
SELECT
    c.clinic_name,
    c.city,
    COUNT(*)                                    AS appointments,
    SUM(a.no_show)                              AS no_shows,
    ROUND(100.0 * SUM(a.no_show) / COUNT(*), 1) AS no_show_pct
FROM appointments a
JOIN clinics c ON a.clinic_id = c.clinic_id
GROUP BY c.clinic_id, c.clinic_name, c.city
HAVING COUNT(*) >= 95
ORDER BY no_show_pct DESC;
"""
print("--- Task 2.2a: Clinics with >= 95 Appointments ---")
print(pd.read_sql_query(q_2_2a, conn).to_string(index=False))

# 2.2b: All 8 clinics (HAVING removed)
q_2_2b = """
SELECT
    c.clinic_name,
    c.city,
    COUNT(*)                                    AS appointments,
    SUM(a.no_show)                              AS no_shows,
    ROUND(100.0 * SUM(a.no_show) / COUNT(*), 1) AS no_show_pct
FROM appointments a
JOIN clinics c ON a.clinic_id = c.clinic_id
GROUP BY c.clinic_id, c.clinic_name, c.city
ORDER BY no_show_pct DESC;
"""
sql_all_clinics = pd.read_sql_query(q_2_2b, conn)
print("\n--- Task 2.2b: All 8 Clinics ---")
print(sql_all_clinics.to_string(index=False))

# Python groupby verification against SQL
appts = pd.read_csv("appointments_clean.csv")
clinics_df = pd.read_csv("clinics_enriched.csv")

py_all_clinics = (
    appts.merge(clinics_df[["clinic_id", "clinic_name", "city"]], on="clinic_id", how="inner")
    .groupby(["clinic_name", "city"], as_index=False)
    .agg(appointments=("appointment_id", "count"), no_shows=("no_show", "sum"))
)
py_all_clinics["no_show_pct"] = (100.0 * py_all_clinics["no_shows"] / py_all_clinics["appointments"]).round(1)
py_all_clinics = py_all_clinics.sort_values("no_show_pct", ascending=False).reset_index(drop=True)

print("\nExact SQL vs Python match:", sql_all_clinics.equals(py_all_clinics));

# Task 2.3  Does booking far ahead matter?

query = """
SELECT
  CASE
    WHEN lead_days <= 7  THEN '1-7'
    WHEN lead_days <= 14 THEN '8-14'
    WHEN lead_days <= 30 THEN '15-30'
    ELSE '31+'
    END                                            AS lead_band,
    COUNT(*)                                       AS appointments,
    SUM(no_show)                                   AS no_shows,
    ROUND(100.0 * SUM(no_show) / COUNT(*), 1)      AS no_show_pct
FROM appointments
GROUP BY lead_band
ORDER BY MIN(lead_days);
"""

# Task 2.4  Month-by-month trend

query = """
WITH monthly AS (
    SELECT
        SUBSTR(appt_date, 1, 7)                    AS month,
        COUNT(*)                                   AS appointments,
        ROUND(100.0 * SUM(no_show) / COUNT(*), 1)  AS no_show_pct
    FROM appointments
    GROUP BY SUBSTR(appt_date, 1, 7)
)
SELECT
    month,
    appointments,
    no_show_pct,
    ROUND(
        no_show_pct - LAG(no_show_pct) OVER (ORDER BY month),
        1
    )                                              AS pct_point_change
FROM monthly
ORDER BY month;
"""

# Task 2.5  Rank clinics inside each city

query = """
WITH clinic_rates AS (
    SELECT
        c.clinic_name                              AS clinic,
        c.city,
        ROUND(100.0 * SUM(a.no_show) / COUNT(*), 1) AS no_show_pct
    FROM appointments a
    JOIN clinics c
        ON a.clinic_id = c.clinic_id
    GROUP BY c.clinic_id, c.clinic_name, c.city
)
SELECT
    clinic,
    city,
    no_show_pct,
    RANK() OVER (PARTITION BY city ORDER BY no_show_pct DESC) AS rank_in_city,
    CASE
        WHEN no_show_pct > (SELECT 100.0 * SUM(no_show) / COUNT(*) FROM appointments) THEN 'above'
        WHEN no_show_pct < (SELECT 100.0 * SUM(no_show) / COUNT(*) FROM appointments) THEN 'below'
        ELSE 'equal'
    END AS vs_overall
FROM clinic_rates
ORDER BY city, rank_in_city;
"""
