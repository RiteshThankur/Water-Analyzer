# water_analyzer.py (Enhanced Full Version)
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.pdfgen import canvas
from io import BytesIO
import smtplib
from email.message import EmailMessage

# ---------- DEFAULT WHO STANDARDS ----------
DEFAULT_WHO_STANDARDS = {
    "pH": (6.5, 8.5),
    "Turbidity (NTU)": (0.0, 5.0),
    "TDS (ppm)": (0.0, 500.0),
    "Temperature (°C)": (5.0, 25.0),
    "Conductivity (µS/cm)": (0.0, 1500.0),
    "Hardness (mg/L CaCO3)": (0.0, 300.0),
}

# ---------- DEFAULT SUGGESTIONS ----------
DEFAULT_SUGGESTIONS = {
    "pH": "Adjust pH by adding acidic or alkaline substances accordingly.",
    "Turbidity (NTU)": "Consider filtration or settling processes to reduce turbidity.",
    "TDS (ppm)": "Use reverse osmosis or distillation to reduce dissolved solids.",
    "Temperature (°C)": "Store water in a cooler place to maintain temperature.",
    "Conductivity (µS/cm)": "High conductivity indicates high dissolved salts; filtration advised.",
    "Hardness (mg/L CaCO3)": "Water softening treatments (ion exchange) can reduce hardness.",
}

# ---------- APP CONFIG ----------
st.set_page_config(page_title="💧 Water Quality Analyzer — Enhanced", layout="wide")
st.title("💧Water Quality Analyzer")
st.markdown("Analyze water parameters, compute a Water Quality Index (WQI), track history, compare reports, export and optionally email PDFs.")

# ---------- SESSION STATE INIT ----------
if "history" not in st.session_state:
    # history: list of dict {time:..., data: {param: value}, status: [...], wqi: float}
    st.session_state.history = []

if "custom_standards" not in st.session_state:
    st.session_state.custom_standards = DEFAULT_WHO_STANDARDS.copy()

if "weights" not in st.session_state:
    # default weights (sum should be ~1.0)
    st.session_state.weights = {
        "pH": 0.2,
        "Turbidity (NTU)": 0.2,
        "TDS (ppm)": 0.15,
        "Temperature (°C)": 0.1,
        "Conductivity (µS/cm)": 0.15,
        "Hardness (mg/L CaCO3)": 0.2,
    }

# ---------- UTILITIES ----------
def calculate_status(data, standards):
    """Return list of 'Safe'/'Unsafe' and list of issue messages."""
    status_list = []
    issues = []
    for p, v in data.items():
        mn, mx = standards[p]
        if not (mn <= v <= mx):
            status_list.append("Unsafe")
            issues.append(f"❌ {p} = {v} (expected {mn} - {mx})")
        else:
            status_list.append("Safe")
    return status_list, issues

def calculate_wqi(values, weights, standards):
    """
    Simple WQI: For each parameter compute score (0-100).
    If within range => 100. Otherwise penalty proportional to relative deviation.
    Weighted average across parameters.
    """
    total_score = 0.0
    total_weight = 0.0
    for p, v in values.items():
        w = weights.get(p, 0)
        mn, mx = standards[p]
        if mn <= v <= mx:
            score = 100.0
        else:
            # compute normalized deviation - avoid division by zero
            if v < mn and mn != 0:
                deviation = (mn - v) / abs(mn)
            elif v > mx and mx != 0:
                deviation = (v - mx) / abs(mx)
            else:
                deviation = 1.0  # fallback large deviation
            score = max(0.0, 100.0 - deviation * 100.0)
        total_score += score * w
        total_weight += w
    if total_weight == 0:
        return 0.0
    return round(total_score / total_weight, 2)

def generate_pdf_report(df, wqi, overall_status, timestamp, notes=""):
    """Return BytesIO containing generated PDF report."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=(595, 842))  # A4-ish
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, 800, "Water Quality Report")
    c.setFont("Helvetica", 11)
    c.drawString(40, 780, f"Date & Time: {timestamp}")
    c.drawString(40, 765, f"WQI Score: {wqi}  |  Overall: {overall_status}")
    if notes:
        c.drawString(40, 748, f"Notes: {notes}")
    y = 720
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Parameters")
    y -= 18
    c.setFont("Helvetica", 11)
    for idx, row in df.iterrows():
        text = f"{row['Parameter']}: {row['Value']} ({row['Status']}) | Range: {row['Safe Range']}"
        if y < 80:
            c.showPage()
            y = 800
            c.setFont("Helvetica", 11)
        c.drawString(40, y, text)
        y -= 16
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def send_email_with_attachment(smtp_host, smtp_port, smtp_user, smtp_pass, to_email, subject, body, attachment_bytes, attachment_name):
    """Send an email with the given bytes attachment. Returns True/False and message."""
    try:
        msg = EmailMessage()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)
        msg.add_attachment(attachment_bytes.getvalue(), maintype="application", subtype="pdf", filename=attachment_name)
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully."
    except Exception as e:
        return False, str(e)

# ---------- SIDEBAR: CONFIGURATION ----------
st.sidebar.header("Configuration")
use_custom = st.sidebar.checkbox("Use custom thresholds (override WHO defaults)", value=False)

# Allow editing custom thresholds if checked
if use_custom:
    st.sidebar.markdown("**Set custom safe ranges** (min, max):")
    for p in DEFAULT_WHO_STANDARDS.keys():
        mn, mx = st.sidebar.slider(p, min_value=float(DEFAULT_WHO_STANDARDS[p][0] - 1000),
                                   max_value=float(DEFAULT_WHO_STANDARDS[p][1] + 1000),
                                   value=(float(DEFAULT_WHO_STANDARDS[p][0]), float(DEFAULT_WHO_STANDARDS[p][1])))
        st.session_state.custom_standards[p] = (mn, mx)
else:
    # keep WHO defaults
    st.session_state.custom_standards = DEFAULT_WHO_STANDARDS.copy()

st.sidebar.markdown("---")
st.sidebar.markdown("**WQI Weight Configuration** (weights influence WQI score)")
for p in DEFAULT_WHO_STANDARDS.keys():
    w = st.sidebar.slider(f"Weight: {p}", min_value=0.0, max_value=1.0, value=float(st.session_state.weights.get(p, 0.15)), step=0.01)
    st.session_state.weights[p] = w

# normalize weights to sum to 1 for interpretation (if all zero, leave)
total_w = sum(st.session_state.weights.values())
if total_w > 0:
    for k in st.session_state.weights:
        st.session_state.weights[k] = st.session_state.weights[k] / total_w

st.sidebar.markdown("---")
st.sidebar.markdown("**Optional: Email PDF Report**")
smtp_host = st.sidebar.text_input("SMTP Host (eg smtp.gmail.com)", value="")
smtp_port = st.sidebar.number_input("SMTP Port", value=587)
smtp_user = st.sidebar.text_input("SMTP Username (email)", value="")
smtp_pass = st.sidebar.text_input("SMTP Password", value="", type="password")
# ---------- TABS ----------
tabs = st.tabs(["🔍 Analyze", "📈 Trends & History", "🧾 Reports", "ℹ️ About"])

# ---------- TAB 1: ANALYZE ----------
with tabs[0]:
    st.header("🔍 Analyze Water Sample")
    st.markdown("Enter measured values for each parameter and click **Analyze**.")

    # input fields (default to mid of standard range)
    inputs = {}
    cols = st.columns(3)
    i = 0
    for p, (mn, mx) in st.session_state.custom_standards.items():
        default_val = round((mn + mx) / 2, 2) if (not np.isnan(mn) and not np.isnan(mx)) else 0.0
        with cols[i % 3]:
            val = st.number_input(label=p, value=float(default_val), format="%.2f", step=0.1)
        inputs[p] = val
        i += 1

    notes = st.text_area("Additional Notes (optional)", value="")
    analyze_btn = st.button("🔍 Analyze Water Quality")

    if analyze_btn:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_list, issues = calculate_status(inputs, st.session_state.custom_standards)
        wqi = calculate_wqi(inputs, st.session_state.weights, st.session_state.custom_standards)
        overall = "Safe" if all(s == "Safe" for s in status_list) else "Unsafe"

        # Save record in history
        st.session_state.history.append({
            "time": timestamp,
            "data": inputs.copy(),
            "status": status_list.copy(),
            "wqi": wqi,
            "notes": notes
        })

        # Display results
        st.subheader("🔬 Summary")
        if overall == "Safe":
            st.success(f"✅ Water is Safe to Drink — WQI = {wqi}")
        else:
            st.error(f"⚠️ Water is Unsafe — WQI = {wqi}")
        if issues:
            st.subheader("⚠️ Issues Detected")
            for issue in issues:
                st.warning(issue)

        # suggestions
        st.subheader("💡 Suggestions")
        for p, s in zip(inputs.keys(), status_list):
            if s == "Unsafe":
                st.info(f"**{p}:** {DEFAULT_SUGGESTIONS.get(p, 'No suggestion available.')}")

        # parameter overview metrics
        st.subheader("📋 Parameter Overview")
        metric_cols = st.columns(len(inputs))
        for idx, (p, v) in enumerate(inputs.items()):
            color = "#16a34a" if status_list[idx] == "Safe" else "#dc2626"
            metric_cols[idx].markdown(f"<div style='text-align:center'><h3 style='color:{color}; margin:0'>{v:.2f}</h3><div style='font-size:12px;color:gray'>{p}</div></div>", unsafe_allow_html=True)

        # detailed table
        df = pd.DataFrame({
            "Parameter": list(inputs.keys()),
            "Value": list(inputs.values()),
            "Safe Range": [f"{st.session_state.custom_standards[p][0]} - {st.session_state.custom_standards[p][1]}" for p in inputs],
            "Status": status_list
        })
        st.subheader("📊 Detailed Table")
        st.dataframe(df, use_container_width=True)

        # visualization
        st.subheader("📈 Visualization")
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['green' if s == "Safe" else 'red' for s in status_list]
        ax.bar(df["Parameter"], df["Value"], color=colors)
        for p in df["Parameter"]:
            mn, mx = st.session_state.custom_standards[p]
            ax.axhline(mn, color='gray', linestyle='--', linewidth=0.8)
            ax.axhline(mx, color='gray', linestyle='--', linewidth=0.8)
        ax.set_ylabel("Value")
        ax.set_title("Parameters vs Safe Range")
        plt.xticks(rotation=30)
        st.pyplot(fig)

        # show WQI breakdown briefly
        st.subheader("🌍 WQI Breakdown")
        st.write(f"**WQI Score:** {wqi}")
        st.write("**Weights used:**")
        st.json(st.session_state.weights)

        # PDF export & optional email
        pdf_buffer = generate_pdf_report(df, wqi, overall, timestamp, notes)
        st.download_button(label="📄 Download PDF Report", data=pdf_buffer, file_name=f"Water_Report_{timestamp.replace(' ', '_').replace(':','-')}.pdf", mime="application/pdf")

        # Email send UI
        if smtp_user and smtp_pass and smtp_host:
            st.markdown("**Email this report**")
            to_email = st.text_input("Recipient Email", value="")
            if st.button("✉️ Send Report by Email"):
                if not to_email:
                    st.error("Please enter recipient email.")
                else:
                    sent, msg = send_email_with_attachment(
                        smtp_host=smtp_host,
                        smtp_port=int(smtp_port),
                        smtp_user=smtp_user,
                        smtp_pass=smtp_pass,
                        to_email=to_email,
                        subject=f"Water Report - {timestamp}",
                        body=f"Please find attached the water quality report generated at {timestamp}. WQI: {wqi} | Overall: {overall}",
                        attachment_bytes=pdf_buffer,
                        attachment_name=f"Water_Report_{timestamp.replace(' ', '_').replace(':','-')}.pdf"
                    )
                    if sent:
                        st.success("Email sent successfully.")
                    else:
                        st.error(f"Failed to send email: {msg}")
        else:
            st.info("Configure SMTP details in the sidebar to enable email sending.")

# ---------- TAB 2: TRENDS & HISTORY ----------
with tabs[1]:
    st.header("📈 Trends & History")
    if not st.session_state.history:
        st.info("No analysis history yet. Run an analysis in the 'Analyze' tab.")
    else:
        # Build history dataframe
        hist_records = []
        for rec in st.session_state.history:
            row = {"Time": rec["time"], "WQI": rec.get("wqi", np.nan)}
            for p, v in rec["data"].items():
                row[p] = v
            hist_records.append(row)
        df_history = pd.DataFrame(hist_records)
        df_history["Time"] = pd.to_datetime(df_history["Time"])

        st.subheader("📈 Parameter Trends Over Time")
        fig, ax = plt.subplots(figsize=(10, 5))
        for p in DEFAULT_WHO_STANDARDS.keys():
            if p in df_history.columns:
                ax.plot(df_history["Time"], df_history[p], marker='o', label=p)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("Parameter Trends")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        st.pyplot(fig)

        st.subheader("🌍 WQI Trend")
        if "WQI" in df_history.columns:
            fig2, ax2 = plt.subplots(figsize=(8, 3.5))
            ax2.plot(df_history["Time"], df_history["WQI"], marker='o')
            ax2.set_xlabel("Time")
            ax2.set_ylabel("WQI")
            ax2.set_title("WQI Over Time")
            plt.xticks(rotation=30)
            st.pyplot(fig2)
        else:
            st.info("WQI scores not available in history yet.")

        st.subheader("📚 History Table")
        # Show reverse chronological
        for rec in reversed(st.session_state.history):
            with st.expander(f"Report at {rec['time']} — WQI: {rec.get('wqi','-')}"):
                rec_df = pd.DataFrame({
                    "Parameter": list(rec["data"].keys()),
                    "Value": list(rec["data"].values()),
                    "Safe Range": [f"{st.session_state.custom_standards[p][0]} - {st.session_state.custom_standards[p][1]}" for p in rec["data"]],
                    "Status": rec["status"]
                })
                st.dataframe(rec_df, use_container_width=True)
                # Download single report PDF in each expander
                pdf_buf = generate_pdf_report(rec_df, rec.get("wqi", 0), "Safe" if all(s == "Safe" for s in rec["status"]) else "Unsafe", rec["time"], rec.get("notes",""))
                st.download_button(label="Download this report (PDF)", data=pdf_buf, file_name=f"Water_Report_{rec['time'].replace(' ', '_').replace(':','-')}.pdf", mime="application/pdf")
        # Export full history to CSV
        if st.button("📥 Download Full History (CSV)"):
            # flatten each record
            flattened = []
            for rec in st.session_state.history:
                row = {"Time": rec["time"], "WQI": rec.get("wqi", "")}
                for p, v in rec["data"].items():
                    row[p] = v
                for idx, stt in enumerate(rec["status"]):
                    # add status columns
                    row[f"{list(rec['data'].keys())[idx]} Status"] = stt
                flattened.append(row)
            full_df = pd.DataFrame(flattened)
            csv_bytes = full_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV", data=csv_bytes, file_name="Water_History.csv", mime="text/csv")

    # Option: Upload CSV to load into history (must match columns)
    st.markdown("---")
    st.subheader("Import History from CSV (optional)")
    uploaded = st.file_uploader("Upload CSV with columns: Time, parameters..., WQI (optional)", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            # minimal validation: must contain Time column
            if "Time" not in uploaded_df.columns:
                st.error("CSV must contain a 'Time' column.")
            else:
                # convert rows into history entries
                for _, row in uploaded_df.iterrows():
                    data = {}
                    status = []
                    for p in DEFAULT_WHO_STANDARDS.keys():
                        if p in uploaded_df.columns:
                            val = float(row[p])
                            data[p] = val
                            mn, mx = st.session_state.custom_standards[p]
                            status.append("Safe" if mn <= val <= mx else "Unsafe")
                    st.session_state.history.append({
                        "time": str(row["Time"]),
                        "data": data,
                        "status": status,
                        "wqi": float(row["WQI"]) if "WQI" in uploaded_df.columns else calculate_wqi(data, st.session_state.weights, st.session_state.custom_standards),
                        "notes": ""
                    })
                st.success("Imported CSV into history successfully.")
        except Exception as e:
            st.error(f"Failed to import CSV: {e}")

# ---------- TAB 3: REPORTS (Comparison + Utilities) ----------
with tabs[2]:
    st.header("🧾 Reports & Comparison")
    if len(st.session_state.history) >= 1:
        st.subheader("Latest Report")
        latest = st.session_state.history[-1]
        latest_df = pd.DataFrame({
            "Parameter": list(latest["data"].keys()),
            "Value": list(latest["data"].values()),
            "Safe Range": [f"{st.session_state.custom_standards[p][0]} - {st.session_state.custom_standards[p][1]}" for p in latest["data"]],
            "Status": latest["status"]
        })
        st.dataframe(latest_df, use_container_width=True)
        st.write(f"WQI: **{latest.get('wqi', '-') }**")
    else:
        st.info("No reports yet. Run analysis first.")

    # Compare last two reports
    if len(st.session_state.history) >= 2:
        st.subheader("📊 Compare Last Two Reports")
        prev = st.session_state.history[-2]
        last = st.session_state.history[-1]
        compare_df = pd.DataFrame({
            "Parameter": list(last["data"].keys()),
            "Previous Value": [prev["data"].get(p, np.nan) for p in last["data"].keys()],
            "Latest Value": [last["data"].get(p, np.nan) for p in last["data"].keys()],
            "Safe Range": [f"{st.session_state.custom_standards[p][0]} - {st.session_state.custom_standards[p][1]}" for p in last["data"].keys()]
        })
        st.dataframe(compare_df, use_container_width=True)

        # show simple delta column
        compare_df["Delta"] = compare_df["Latest Value"] - compare_df["Previous Value"]
        st.subheader("Comparison Deltas")
        st.dataframe(compare_df[["Parameter", "Previous Value", "Latest Value", "Delta"]], use_container_width=True)
    else:
        st.info("Need at least two reports to compare.")

    # Option to clear history
    st.markdown("---")
    if st.button("⚠️ Clear All History (irreversible)"):
        st.session_state.history = []
        st.success("History cleared.")

# ---------- TAB 4: ABOUT ----------
with tabs[3]:
    st.header("ℹ️ About This App")
    st.markdown("""
    **Enhanced Water Quality Analyzer**
    - Computes per-parameter safety checks against WHO or custom thresholds.
    - Calculates a weighted Water Quality Index (WQI) and shows trends.
    - Export single reports to PDF and entire history to CSV.
    - Optional: Email PDFs directly (configure SMTP in sidebar).
    """)
    st.markdown("**Parameters monitored:**")
    for p, (mn, mx) in DEFAULT_WHO_STANDARDS.items():
        st.markdown(f"- **{p}**: {mn} — {mx}")
    st.markdown("---")
    st.markdown("**Developer:** You (enhanced by ChatGPT)\n\nFeel free to ask for further features: mapping (folium), ML prediction, or a Streamlit dashboard with interactive maps.")
