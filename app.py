"""
CyberShield AI — Network Intrusion Detection System
Author: Abhay Sharma | github.com/KAZURIKAFU
ML-powered real-time network threat detection dashboard
Random Forest | SVM | Neural Network
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from ml_models import MODELS
from traffic_simulator import (generate_batch, generate_historical_traffic,
                                ATTACK_TYPES)
from threat_analyzer import analyze_batch, SEVERITY_MAP

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="CyberShield AI | Abhay Sharma",
                meta_tags=[{"name":"viewport","content":"width=device-width,initial-scale=1"}],
                suppress_callback_exceptions=True)

C = {"bg":"#0a0e1a","card":"#0d1117","card2":"#161b22","card3":"#1c2128",
     "blue":"#58a6ff","green":"#3fb950","orange":"#f78166","yellow":"#e3b341",
     "purple":"#bc8cff","red":"#ff4444","text":"#e6edf3","sub":"#8b949e",
     "border":"#21262d","accent":"#1f6feb"}

# ── Pre-generate historical data ──────────────────────────────────────────────
HIST_DF = generate_historical_traffic(24)

# ── Helpers ───────────────────────────────────────────────────────────────────
LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              font=dict(color=C["text"], family="Segoe UI, Arial"),
              margin=dict(l=10, r=10, t=35, b=30),
              xaxis=dict(showgrid=False, color=C["sub"], tickfont=dict(size=10)),
              yaxis=dict(showgrid=True, gridcolor=C["border"], color=C["sub"],
                         tickfont=dict(size=10)),
              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["sub"], size=10)))

ATK_COLORS = {k: v["color"] for k, v in ATTACK_TYPES.items()}


def kpi_card(title, value, subtitle, icon, color, width="1fr"):
    return html.Div(style={
        "backgroundColor": C["card2"], "borderRadius": "10px",
        "padding": "16px 18px", "border": f"1px solid {C['border']}",
        "borderTop": f"3px solid {color}"}, children=[
        html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"flex-start"}, children=[
            html.Div(children=[
                html.P(title, style={"margin":"0 0 6px 0","fontSize":"10px","color":C["sub"],
                                     "textTransform":"uppercase","letterSpacing":"1px"}),
                html.H2(str(value), style={"margin":"0","fontSize":"26px","fontWeight":"700",
                                           "color":C["text"]}),
                html.P(subtitle, style={"margin":"4px 0 0 0","fontSize":"11px","color":color}),
            ]),
            html.Span(icon, style={"fontSize":"28px"}),
        ])
    ])


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(style={"backgroundColor":C["bg"],"minHeight":"100vh",
                              "fontFamily":"'Segoe UI',Arial,sans-serif","color":C["text"]}, children=[

    # Header
    html.Div(style={"background":"linear-gradient(135deg,#0a0e1a 0%,#0d1f3c 50%,#0a1628 100%)",
                    "padding":"18px 36px","borderBottom":f"2px solid {C['accent']}",
                    "boxShadow":"0 2px 20px rgba(88,166,255,0.15)"}, children=[
        html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center"}, children=[
            html.Div(style={"display":"flex","alignItems":"center","gap":"14px"}, children=[
                html.Span("🛡️", style={"fontSize":"32px"}),
                html.Div(children=[
                    html.H1("CyberShield AI",
                            style={"margin":"0","fontSize":"22px","fontWeight":"700",
                                   "color":"#ffffff","letterSpacing":"-0.3px"}),
                    html.P("Network Intrusion Detection System · ML-Powered · Real-Time",
                           style={"margin":"2px 0 0 0","fontSize":"11px","color":C["blue"]}),
                ])
            ]),
            html.Div(style={"display":"flex","gap":"8px","alignItems":"center"}, children=[
                html.Div(id="live-status",
                         style={"display":"flex","alignItems":"center","gap":"6px",
                                "backgroundColor":"rgba(63,185,80,0.1)",
                                "border":f"1px solid {C['green']}","borderRadius":"20px",
                                "padding":"4px 12px"}, children=[
                    html.Div(style={"width":"8px","height":"8px","borderRadius":"50%",
                                    "backgroundColor":C["green"]}),
                    html.Span("LIVE", style={"fontSize":"11px","color":C["green"],"fontWeight":"700"}),
                ]),
                html.Div(id="clock", style={"fontSize":"12px","color":C["sub"]}),
                html.Div(style={"display":"flex","gap":"6px"}, children=[
                    html.Span(f"🌲 RF: {MODELS.metrics['Random Forest']['accuracy']}%",
                              style={"backgroundColor":"rgba(88,166,255,0.1)","color":C["blue"],
                                     "padding":"3px 8px","borderRadius":"10px","fontSize":"10px",
                                     "border":f"1px solid {C['blue']}"}),
                    html.Span(f"⚡ SVM: {MODELS.metrics['SVM']['accuracy']}%",
                              style={"backgroundColor":"rgba(188,140,255,0.1)","color":C["purple"],
                                     "padding":"3px 8px","borderRadius":"10px","fontSize":"10px",
                                     "border":f"1px solid {C['purple']}"}),
                    html.Span(f"🧠 NN: {MODELS.metrics['Neural Network']['accuracy']}%",
                              style={"backgroundColor":"rgba(63,185,80,0.1)","color":C["green"],
                                     "padding":"3px 8px","borderRadius":"10px","fontSize":"10px",
                                     "border":f"1px solid {C['green']}"}),
                ]),
                html.Div(style={"fontSize":"11px","color":C["sub"]}, children=[
                    html.Span("By ", style={"color":C["sub"]}),
                    html.Span("Abhay Sharma", style={"color":C["blue"],"fontWeight":"600"}),
                ]),
            ])
        ])
    ]),

    dcc.Interval(id="interval", interval=2500, n_intervals=0),
    dcc.Store(id="packet-store", data=[]),
    dcc.Store(id="stats-store", data={"total":0,"threats":0,"blocked":0,"critical":0}),

    # Body
    html.Div(style={"padding":"20px 36px"}, children=[

        # KPI Row
        html.Div(id="kpi-row",
                 style={"display":"grid","gridTemplateColumns":"repeat(5,1fr)",
                        "gap":"14px","marginBottom":"20px"}),

        # Row 1: Live Traffic + Attack Distribution
        html.Div(style={"display":"grid","gridTemplateColumns":"2fr 1fr",
                        "gap":"14px","marginBottom":"14px"}, children=[
            html.Div(style={"backgroundColor":C["card2"],"borderRadius":"10px",
                             "padding":"16px","border":f"1px solid {C['border']}"}, children=[
                html.H3("📡 Live Network Traffic",
                        style={"margin":"0 0 12px 0","fontSize":"13px","color":C["blue"]}),
                dcc.Graph(id="live-traffic", style={"height":"240px"},
                          config={"displayModeBar":False}),
            ]),
            html.Div(style={"backgroundColor":C["card2"],"borderRadius":"10px",
                             "padding":"16px","border":f"1px solid {C['border']}"}, children=[
                html.H3("⚔️ Attack Distribution",
                        style={"margin":"0 0 12px 0","fontSize":"13px","color":C["orange"]}),
                dcc.Graph(id="attack-pie", style={"height":"240px"},
                          config={"displayModeBar":False}),
            ]),
        ]),

        # Row 2: Historical Trend + Model Comparison
        html.Div(style={"display":"grid","gridTemplateColumns":"3fr 2fr",
                        "gap":"14px","marginBottom":"14px"}, children=[
            html.Div(style={"backgroundColor":C["card2"],"borderRadius":"10px",
                             "padding":"16px","border":f"1px solid {C['border']}"}, children=[
                html.H3("📈 24-Hour Attack Timeline",
                        style={"margin":"0 0 12px 0","fontSize":"13px","color":C["purple"]}),
                dcc.Graph(id="timeline-chart", style={"height":"240px"},
                          config={"displayModeBar":False}),
            ]),
            html.Div(style={"backgroundColor":C["card2"],"borderRadius":"10px",
                             "padding":"16px","border":f"1px solid {C['border']}"}, children=[
                html.H3("🤖 ML Model Comparison",
                        style={"margin":"0 0 12px 0","fontSize":"13px","color":C["green"]}),
                dcc.Graph(id="model-comparison", style={"height":"240px"},
                          config={"displayModeBar":False}),
            ]),
        ]),

        # Row 3: Alerts + Packet Log + Top Attackers
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 2fr 1fr",
                        "gap":"14px","marginBottom":"14px"}, children=[

            # Live Alerts
            html.Div(style={"backgroundColor":C["card2"],"borderRadius":"10px",
                             "padding":"16px","border":f"1px solid {C['border']}"}, children=[
                html.H3("🚨 Live Alerts",
                        style={"margin":"0 0 12px 0","fontSize":"13px","color":C["red"]}),
                html.Div(id="alerts-panel",
                         style={"maxHeight":"280px","overflowY":"auto"}),
            ]),

            # Packet Log
            html.Div(style={"backgroundColor":C["card2"],"borderRadius":"10px",
                             "padding":"16px","border":f"1px solid {C['border']}"}, children=[
                html.H3("📋 Live Packet Log",
                        style={"margin":"0 0 12px 0","fontSize":"13px","color":C["yellow"]}),
                html.Div(id="packet-log",
                         style={"maxHeight":"280px","overflowY":"auto","fontSize":"11px"}),
            ]),

            # Top Attackers + Protocol
            html.Div(style={"backgroundColor":C["card2"],"borderRadius":"10px",
                             "padding":"16px","border":f"1px solid {C['border']}"}, children=[
                html.H3("🌍 Top Attack Sources",
                        style={"margin":"0 0 12px 0","fontSize":"13px","color":C["orange"]}),
                html.Div(id="top-attackers"),
                html.Hr(style={"border":f"1px solid {C['border']}","margin":"12px 0"}),
                html.H3("🔌 Protocol Mix",
                        style={"margin":"0 0 8px 0","fontSize":"13px","color":C["blue"]}),
                dcc.Graph(id="protocol-chart", style={"height":"120px"},
                          config={"displayModeBar":False}),
            ]),
        ]),

        # Footer
        html.Div(style={"textAlign":"center","padding":"12px 0",
                         "borderTop":f"1px solid {C['border']}","color":C["sub"],
                         "fontSize":"11px"}, children=[
            html.P("CyberShield AI · Built by Abhay Sharma · Manipal University Jaipur · "
                   "Cyber Security Certified (NSDC + Nimesa/Launched Global) · "
                   "github.com/KAZURIKAFU",
                   style={"margin":"0"})
        ]),
    ])
])


# ── Main Callback ─────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-row","children"),
    Output("live-traffic","figure"),
    Output("attack-pie","figure"),
    Output("timeline-chart","figure"),
    Output("model-comparison","figure"),
    Output("alerts-panel","children"),
    Output("packet-log","children"),
    Output("top-attackers","children"),
    Output("protocol-chart","figure"),
    Output("packet-store","data"),
    Output("stats-store","data"),
    Output("clock","children"),
    Input("interval","n_intervals"),
    State("packet-store","data"),
    State("stats-store","data"),
)
def update_dashboard(n, stored_packets, stored_stats):
    from datetime import datetime

    # Generate & predict new packets
    new_df = generate_batch(15, n)
    new_df = MODELS.predict_batch(new_df, "Random Forest")
    new_records = new_df.to_dict("records")

    # Rolling window of last 200 packets
    all_packets = (stored_packets or []) + new_records
    all_packets = all_packets[-200:]
    df = pd.DataFrame(all_packets)

    # Update cumulative stats
    stats = stored_stats or {"total":0,"threats":0,"blocked":0,"critical":0}
    stats["total"]    += len(new_df)
    stats["threats"]  += int(new_df["is_threat"].sum())
    stats["blocked"]  += int(new_df["is_threat"].sum() * 0.92)
    stats["critical"] += int((new_df["severity"] >= 3).sum())

    analysis = analyze_batch(df)
    score    = analysis.get("security_score", 95)
    grade    = analysis.get("security_grade", {"grade":"A","label":"Excellent","color":C["green"]})

    # ── KPI Cards ──
    kpis = [
        kpi_card("Total Packets",   f"{stats['total']:,}",
                 "packets analyzed", "📦", C["blue"]),
        kpi_card("Threats Detected",f"{stats['threats']:,}",
                 f"{analysis.get('threat_rate',0)}% threat rate", "⚔️", C["orange"]),
        kpi_card("Threats Blocked", f"{stats['blocked']:,}",
                 "92% block rate", "🛡️", C["green"]),
        kpi_card("Critical Alerts", f"{stats['critical']:,}",
                 "High/Critical severity", "🚨", C["red"]),
        kpi_card("Security Score",  f"{score}/100",
                 f"Grade {grade['grade']} — {grade['label']}", "🔒", grade["color"]),
    ]

    # ── Live Traffic ──
    threat_counts = df.groupby(["timestamp","attack_type"]).size().reset_index(name="count")
    fig_traffic = go.Figure()
    for atk, color in ATK_COLORS.items():
        sub = threat_counts[threat_counts["attack_type"] == atk]
        if not sub.empty:
            fig_traffic.add_trace(go.Scatter(
                x=sub["timestamp"], y=sub["count"], name=atk,
                mode="lines", stackgroup="one",
                line=dict(color=color, width=1),
                fillcolor=color.replace("#","rgba(").rstrip(")") if "#" in color else color,
            ))
    fig_traffic.update_layout(**LAYOUT, height=240,
                               title=dict(text="Packets/Interval by Type",
                                          font=dict(size=12,color=C["sub"])))

    # ── Attack Pie ──
    atk_counts = df["attack_type"].value_counts().reset_index()
    atk_counts.columns = ["attack","count"]
    fig_pie = go.Figure(go.Pie(
        labels=atk_counts["attack"], values=atk_counts["count"],
        hole=0.5,
        marker=dict(colors=[ATK_COLORS.get(a, C["blue"]) for a in atk_counts["attack"]],
                    line=dict(color=C["bg"], width=2)),
        textfont=dict(color=C["text"], size=10),
    ))
    fig_pie.update_layout(**LAYOUT, height=240, showlegend=True,
                           title=dict(text="Current Window",
                                      font=dict(size=12,color=C["sub"])))

    # ── Timeline ──
    timeline = HIST_DF[HIST_DF["attack_type"] != "Normal"]
    timeline_grp = timeline.groupby(["hour","attack_type"])["count"].sum().reset_index()
    fig_timeline = go.Figure()
    for atk, color in ATK_COLORS.items():
        if atk == "Normal": continue
        sub = timeline_grp[timeline_grp["attack_type"] == atk]
        if not sub.empty:
            fig_timeline.add_trace(go.Bar(
                x=sub["hour"], y=sub["count"], name=atk,
                marker_color=color, opacity=0.85,
            ))
    fig_timeline.update_layout(**LAYOUT, height=240, barmode="stack",
                                title=dict(text="Attacks by Hour (24h)",
                                           font=dict(size=12,color=C["sub"])))

    # ── Model Comparison ──
    metrics_df = MODELS.get_metrics_df()
    fig_models = go.Figure()
    metrics_cols = ["Accuracy","Precision","Recall","F1-Score"]
    bar_colors   = [C["blue"], C["purple"], C["green"], C["yellow"]]
    for i, metric in enumerate(metrics_cols):
        vals = [float(str(v).replace("%","")) for v in metrics_df[metric]]
        fig_models.add_trace(go.Bar(
            name=metric, x=metrics_df["Model"], y=vals,
            marker_color=bar_colors[i], opacity=0.85,
            text=[f"{v:.1f}%" for v in vals], textposition="outside",
            textfont=dict(size=9, color=C["text"]),
        ))
    fig_models.update_layout(**LAYOUT, height=240, barmode="group",
                              yaxis=dict(range=[0,115], showgrid=True,
                                         gridcolor=C["border"], color=C["sub"]),
                              title=dict(text="Accuracy / Precision / Recall / F1",
                                         font=dict(size=12,color=C["sub"])))

    # ── Alerts ──
    alerts = analysis.get("alerts", [])
    alert_elements = []
    for a in alerts:
        alert_elements.append(
            html.Div(style={"padding":"8px 10px","marginBottom":"6px","borderRadius":"6px",
                             "backgroundColor":a["bg"],"border":f"1px solid {a['color']}"}, children=[
                html.Div(style={"display":"flex","justifyContent":"space-between"}, children=[
                    html.Span(a["type"], style={"fontSize":"11px","fontWeight":"700","color":a["color"]}),
                    html.Span(a["severity"], style={"fontSize":"9px","color":a["color"],
                                                     "border":f"1px solid {a['color']}",
                                                     "padding":"1px 5px","borderRadius":"10px"}),
                ]),
                html.P(f"{a['src_ip']} · {a['country']}",
                       style={"margin":"2px 0 0 0","fontSize":"10px","color":C["sub"]}),
                html.P(a["time"], style={"margin":"1px 0 0 0","fontSize":"9px","color":C["sub"]}),
            ])
        )
    if not alert_elements:
        alert_elements = [html.P("✅ No threats detected",
                                  style={"color":C["green"],"fontSize":"12px","margin":"0"})]

    # ── Packet Log ──
    log_rows = []
    header = html.Div(style={"display":"grid",
                               "gridTemplateColumns":"70px 110px 80px 80px 70px 90px",
                               "gap":"4px","padding":"4px 6px","marginBottom":"4px",
                               "borderBottom":f"1px solid {C['border']}"}, children=[
        html.Span(col, style={"color":C["sub"],"fontSize":"10px","fontWeight":"600"})
        for col in ["Time","Src IP","Protocol","Service","Port","Attack"]
    ])
    log_rows.append(header)
    recent = df.tail(15).iloc[::-1]
    for _, row in recent.iterrows():
        color = ATK_COLORS.get(row.get("attack_type","Normal"), C["sub"])
        log_rows.append(
            html.Div(style={"display":"grid",
                             "gridTemplateColumns":"70px 110px 80px 80px 70px 90px",
                             "gap":"4px","padding":"4px 6px",
                             "borderBottom":f"1px solid {C['border']}",
                             "backgroundColor":"rgba(255,255,255,0.02)"}, children=[
                html.Span(str(row.get("timestamp",""))[:8],
                          style={"color":C["sub"],"fontSize":"10px"}),
                html.Span(str(row.get("src_ip","")),
                          style={"color":C["text"],"fontSize":"10px"}),
                html.Span(str(row.get("protocol","")),
                          style={"color":C["blue"],"fontSize":"10px"}),
                html.Span(str(row.get("service","")),
                          style={"color":C["sub"],"fontSize":"10px"}),
                html.Span(str(row.get("dst_port","")),
                          style={"color":C["sub"],"fontSize":"10px"}),
                html.Span(str(row.get("attack_type",""))[:12],
                          style={"color":color,"fontSize":"10px","fontWeight":"600"}),
            ])
        )

    # ── Top Attackers ──
    top_atk = analysis.get("top_attackers", {})
    max_count = max(top_atk.values(), default=1)
    attacker_elements = []
    for country, count in list(top_atk.items())[:5]:
        pct = int(count / max_count * 100)
        attacker_elements.append(
            html.Div(style={"marginBottom":"8px"}, children=[
                html.Div(style={"display":"flex","justifyContent":"space-between",
                                 "marginBottom":"2px"}, children=[
                    html.Span(country, style={"fontSize":"11px","color":C["text"]}),
                    html.Span(str(count), style={"fontSize":"10px","color":C["sub"]}),
                ]),
                html.Div(style={"backgroundColor":C["card3"],"borderRadius":"4px","height":"4px"}, children=[
                    html.Div(style={"backgroundColor":C["orange"],"borderRadius":"4px",
                                     "height":"4px","width":f"{pct}%"})
                ])
            ])
        )
    if not attacker_elements:
        attacker_elements = [html.P("No external threats", style={"color":C["green"],
                                                                    "fontSize":"11px","margin":"0"})]

    # ── Protocol Chart ──
    protocols = analysis.get("protocols", {})
    proto_colors = [C["blue"],C["green"],C["yellow"],C["orange"],C["purple"],
                    C["red"],"#79c0ff","#56d364"]
    fig_proto = go.Figure(go.Pie(
        labels=list(protocols.keys()), values=list(protocols.values()),
        hole=0.4,
        marker=dict(colors=proto_colors[:len(protocols)],
                    line=dict(color=C["bg"],width=1)),
        textfont=dict(size=9, color=C["text"]),
        showlegend=False,
    ))
    fig_proto.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=0,r=0,t=0,b=0), height=120)

    clock = datetime.now().strftime("%H:%M:%S")

    return (kpis, fig_traffic, fig_pie, fig_timeline, fig_models,
            alert_elements, log_rows, attacker_elements, fig_proto,
            all_packets, stats, clock)


if __name__ == "__main__":
    print("\n🛡️  CyberShield AI Starting...")
    print("🌐 Open: http://localhost:8052\n")
    app.run(debug=False, port=8052)
