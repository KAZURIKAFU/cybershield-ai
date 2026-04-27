"""
Traffic Simulator — CyberShield AI
Author: Abhay Sharma | github.com/KAZURIKAFU
Simulates realistic network traffic with embedded attack patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ── Attack Definitions ────────────────────────────────────────────────────────
ATTACK_TYPES = {
    "Normal":        {"severity": 0, "color": "#3fb950", "label": "🟢 Normal",      "weight": 0.60},
    "Port Scan":     {"severity": 1, "color": "#e3b341", "label": "🟡 Port Scan",   "weight": 0.12},
    "Brute Force":   {"severity": 2, "color": "#f78166", "label": "🟠 Brute Force", "weight": 0.10},
    "SQL Injection": {"severity": 3, "color": "#ff4444", "label": "🔴 SQL Inject",  "weight": 0.08},
    "DoS":           {"severity": 3, "color": "#ff4444", "label": "🔴 DoS Attack",  "weight": 0.06},
    "DDoS":          {"severity": 4, "color": "#ff0000", "label": "🚨 DDoS Attack", "weight": 0.04},
}

PROTOCOLS   = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "FTP", "SSH", "DNS"]
COUNTRIES   = ["China", "Russia", "USA", "Brazil", "India", "Germany",
               "Netherlands", "Ukraine", "Romania", "Iran", "North Korea", "Unknown"]
SERVICES    = ["http", "ftp", "ssh", "smtp", "dns", "pop3", "imap", "telnet", "https", "mysql"]
FLAGS       = ["SF", "S0", "REJ", "RSTO", "SH", "RSTR", "S1", "S2", "S3", "OTH"]

# ── Feature Engineering ───────────────────────────────────────────────────────
def _packet_features(attack_type: str) -> dict:
    """Generate realistic NSL-KDD style features per attack type."""
    base = {
        "Normal":        {"duration":(0,300),  "src_bytes":(100,50000),  "dst_bytes":(100,50000),
                          "land":0, "wrong_fragment":(0,1),   "urgent":(0,1),
                          "hot":(0,5),           "num_failed_logins":(0,1),
                          "num_compromised":(0,2),"count":(1,100),        "srv_count":(1,100),
                          "serror_rate":(0,0.1), "same_srv_rate":(0.8,1.0)},
        "Port Scan":     {"duration":(0,2),    "src_bytes":(0,100),      "dst_bytes":(0,50),
                          "land":0, "wrong_fragment":(0,0),   "urgent":(0,0),
                          "hot":(0,1),           "num_failed_logins":(0,0),
                          "num_compromised":(0,0),"count":(100,511),      "srv_count":(1,10),
                          "serror_rate":(0.8,1.0),"same_srv_rate":(0.0,0.1)},
        "Brute Force":   {"duration":(1,60),   "src_bytes":(100,1000),   "dst_bytes":(100,500),
                          "land":0, "wrong_fragment":(0,0),   "urgent":(0,0),
                          "hot":(0,2),           "num_failed_logins":(3,20),
                          "num_compromised":(0,1),"count":(50,200),       "srv_count":(50,200),
                          "serror_rate":(0.5,0.9),"same_srv_rate":(0.9,1.0)},
        "SQL Injection": {"duration":(1,30),   "src_bytes":(500,5000),   "dst_bytes":(100,1000),
                          "land":0, "wrong_fragment":(0,2),   "urgent":(0,1),
                          "hot":(5,30),          "num_failed_logins":(0,2),
                          "num_compromised":(1,10),"count":(1,50),        "srv_count":(1,20),
                          "serror_rate":(0.1,0.4),"same_srv_rate":(0.6,0.9)},
        "DoS":           {"duration":(0,5),    "src_bytes":(0,500),      "dst_bytes":(0,100),
                          "land":0, "wrong_fragment":(0,3),   "urgent":(0,0),
                          "hot":(0,1),           "num_failed_logins":(0,0),
                          "num_compromised":(0,0),"count":(200,511),      "srv_count":(200,511),
                          "serror_rate":(0.9,1.0),"same_srv_rate":(0.9,1.0)},
        "DDoS":          {"duration":(0,2),    "src_bytes":(0,200),      "dst_bytes":(0,50),
                          "land":0, "wrong_fragment":(0,5),   "urgent":(0,2),
                          "hot":(0,1),           "num_failed_logins":(0,0),
                          "num_compromised":(0,0),"count":(400,511),      "srv_count":(400,511),
                          "serror_rate":(0.95,1.0),"same_srv_rate":(0.95,1.0)},
    }
    b = base[attack_type]
    result = {}
    for k, v in b.items():
        if isinstance(v, tuple) and all(isinstance(x, int) for x in v):
            lo, hi = v
            result[k] = int(np.random.randint(lo, max(lo+1, hi+1)))
        elif isinstance(v, tuple):
            lo, hi = v
            result[k] = round(float(np.random.uniform(lo, max(lo+0.001, hi))), 3)
        else:
            result[k] = v
    return result


def generate_packet(tick: int = 0) -> dict:
    """Generate a single simulated network packet."""
    weights = [ATTACK_TYPES[a]["weight"] for a in ATTACK_TYPES]
    attack_type = random.choices(list(ATTACK_TYPES.keys()), weights=weights)[0]
    features = _packet_features(attack_type)

    # Simulate IP addresses
    if attack_type == "Normal":
        src_ip = f"192.168.{random.randint(1,10)}.{random.randint(1,254)}"
        country = "Local"
    else:
        src_ip = f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        country = random.choice(COUNTRIES)

    dst_ip   = f"10.0.{random.randint(0,5)}.{random.randint(1,50)}"
    protocol = random.choice(PROTOCOLS)
    service  = random.choice(SERVICES)
    flag     = random.choice(FLAGS)
    src_port = random.randint(1024, 65535)
    dst_port = random.choice([80, 443, 22, 21, 3306, 5432, 8080, 25, 53, 3389])

    return {
        "timestamp":      datetime.now().strftime("%H:%M:%S"),
        "src_ip":         src_ip,
        "dst_ip":         dst_ip,
        "src_port":       src_port,
        "dst_port":       dst_port,
        "protocol":       protocol,
        "service":        service,
        "flag":           flag,
        "country":        country,
        "attack_type":    attack_type,
        "severity":       ATTACK_TYPES[attack_type]["severity"],
        "color":          ATTACK_TYPES[attack_type]["color"],
        **features,
    }


def generate_batch(n: int = 20, tick: int = 0) -> pd.DataFrame:
    """Generate a batch of network packets."""
    return pd.DataFrame([generate_packet(tick) for _ in range(n)])


def generate_historical_traffic(hours: int = 24) -> pd.DataFrame:
    """Generate 24 hours of historical traffic data."""
    rows = []
    start = datetime.now() - timedelta(hours=hours)
    for minute in range(hours * 60):
        ts = start + timedelta(minutes=minute)
        hour = ts.hour
        # Traffic peaks during business hours
        intensity = 1 + 1.5 * abs(np.sin(np.pi * hour / 12))
        n_packets = int(np.random.poisson(50 * intensity))
        for attack_type, info in ATTACK_TYPES.items():
            count = max(0, int(n_packets * info["weight"] * np.random.uniform(0.5, 1.5)))
            if count > 0:
                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                    "hour": hour,
                    "minute": minute,
                    "attack_type": attack_type,
                    "count": count,
                    "severity": info["severity"],
                })
    return pd.DataFrame(rows)


def get_feature_columns() -> list:
    return ["duration", "src_bytes", "dst_bytes", "land", "wrong_fragment",
            "urgent", "hot", "num_failed_logins", "num_compromised",
            "count", "srv_count", "serror_rate", "same_srv_rate"]
