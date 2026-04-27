"""
Threat Analyzer — CyberShield AI
Author: Abhay Sharma | github.com/KAZURIKAFU
Analyzes network packets and generates threat intelligence
"""

import pandas as pd
import numpy as np
from traffic_simulator import ATTACK_TYPES

# ── Severity Labels ───────────────────────────────────────────────────────────
SEVERITY_MAP = {
    0: {"label": "None",     "color": "#3fb950", "bg": "rgba(63,185,80,0.15)"},
    1: {"label": "Low",      "color": "#e3b341", "bg": "rgba(227,179,65,0.15)"},
    2: {"label": "Medium",   "color": "#f78166", "bg": "rgba(247,129,102,0.15)"},
    3: {"label": "High",     "color": "#ff4444", "bg": "rgba(255,68,68,0.15)"},
    4: {"label": "Critical", "color": "#ff0000", "bg": "rgba(255,0,0,0.2)"},
}


def calculate_security_score(threat_counts: dict, total: int) -> int:
    """Calculate overall network security score (0–100)."""
    if total == 0:
        return 100
    weights = {"Normal": 0, "Port Scan": 5, "Brute Force": 15,
               "SQL Injection": 25, "DoS": 25, "DDoS": 40}
    threat_score = sum(threat_counts.get(k, 0) * w for k, w in weights.items())
    normalized = min(100, (threat_score / max(total, 1)) * 100)
    score = max(0, int(100 - normalized))
    return score


def get_security_grade(score: int) -> dict:
    """Convert security score to letter grade."""
    if score >= 90:
        return {"grade": "A", "label": "Excellent", "color": "#3fb950"}
    elif score >= 75:
        return {"grade": "B", "label": "Good",      "color": "#79c0ff"}
    elif score >= 60:
        return {"grade": "C", "label": "Fair",      "color": "#e3b341"}
    elif score >= 40:
        return {"grade": "D", "label": "Poor",      "color": "#f78166"}
    else:
        return {"grade": "F", "label": "Critical",  "color": "#ff4444"}


def analyze_batch(df: pd.DataFrame) -> dict:
    """Full threat analysis of a packet batch."""
    if df.empty:
        return {}

    total = len(df)
    threat_counts = df["attack_type"].value_counts().to_dict()
    threats_only  = df[df["attack_type"] != "Normal"]
    n_threats     = len(threats_only)
    n_blocked     = int(n_threats * 0.92)  # 92% block rate
    n_critical    = len(df[df["severity"] >= 3])

    score = calculate_security_score(threat_counts, total)
    grade = get_security_grade(score)

    # Top attacking countries
    attackers = (threats_only["country"].value_counts().head(5).to_dict()
                 if not threats_only.empty else {})

    # Protocol distribution
    protocols = df["protocol"].value_counts().to_dict()

    # Alerts for high severity
    alerts = []
    for _, row in df[df["severity"] >= 2].iterrows():
        sev = SEVERITY_MAP[row["severity"]]
        alerts.append({
            "time":     row["timestamp"],
            "type":     row["attack_type"],
            "src_ip":   row["src_ip"],
            "country":  row["country"],
            "severity": sev["label"],
            "color":    sev["color"],
            "bg":       sev["bg"],
        })

    return {
        "total_packets":   total,
        "total_threats":   n_threats,
        "blocked":         n_blocked,
        "critical":        n_critical,
        "threat_counts":   threat_counts,
        "security_score":  score,
        "security_grade":  grade,
        "top_attackers":   attackers,
        "protocols":       protocols,
        "alerts":          alerts[:8],
        "threat_rate":     round(n_threats / max(total, 1) * 100, 1),
    }


def format_alert_message(alert: dict) -> str:
    """Format a threat alert into a readable message."""
    return (f"[{alert['time']}] {alert['severity'].upper()} — "
            f"{alert['type']} from {alert['src_ip']} ({alert['country']})")
