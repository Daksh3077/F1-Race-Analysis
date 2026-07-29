import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from openf1_loader import (
    get_seasons, get_races, get_session_key,
    get_drivers, get_laps, get_positions,
    get_pit_stops, get_weather, get_telemetry,
    build_lap_features
)
from predict_upcoming_race import render_prediction_tab

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="F1 Race Analytics",
    page_icon="🏎️",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"] { background-color: #0a0a0a; color: white; }
.stApp { background-color: #0a0a0a; }
h1, h2, h3, h4 { color: #ff1e00; font-family: sans-serif; }
section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 2px solid #ff1e00;
}
div[data-testid="metric-container"] {
    background-color: #151515;
    border: 2px solid #ff1e00;
    padding: 20px;
    border-radius: 15px;
}
.stDataFrame { border: 1px solid #ff1e00; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>🏎️ F1 RACE ANALYTICS</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# =====================================================
# TABS
# =====================================================
tab_live, tab_predict = st.tabs(
    ["📊 Race Analysis", "🔮 Predict Upcoming Race"]
)

with tab_live:

    # =====================================================
    # SIDEBAR
    # =====================================================
    st.sidebar.title("⚙️ Race Settings")
    st.sidebar.caption("Data available: 2023 onwards via OpenF1 API")

    year = st.sidebar.selectbox("Season", get_seasons(), index=1)

    with st.spinner("Fetching race calendar..."):
        races = get_races(year)

    if not races:
        st.error("Could not load race calendar. Please try again.")
        st.stop()

    race_options = {
        f"{r['location']} — {r['country_name']}": r["location"]
        for r in races
    }
    race_label    = st.sidebar.selectbox("Grand Prix", list(race_options.keys()))
    race_location = race_options[race_label]

    # =====================================================
    # LOAD SESSION DATA
    # =====================================================
    session_key = get_session_key(year, race_location)

    if session_key is None:
        st.error(f"Could not find session for {race_label} {year}.")
        st.stop()

    with st.spinner(f"Loading {race_label} {year} data..."):
        drivers_df = get_drivers(session_key)
        laps_df    = get_laps(session_key)
        weather_df = get_weather(session_key)
        pits_df    = get_pit_stops(session_key)

    if laps_df.empty:
        st.error("No lap data available for this race.")
        st.stop()

    df = build_lap_features(laps_df, drivers_df)

    # =====================================================
    # DRIVER LIST
    # =====================================================
    all_drivers = sorted(df["Driver"].dropna().unique().tolist()) \
                  if "Driver" in df.columns else []

    if not all_drivers:
        st.error("No driver data available.")
        st.stop()

    st.sidebar.markdown("---")
    driver_choice = st.sidebar.selectbox(
        "Telemetry Driver", all_drivers, index=0
    )

    # =====================================================
    # GET DRIVER NUMBER for telemetry
    # =====================================================
    def get_driver_number(driver_abbr):
        if drivers_df.empty:
            return None
        row = drivers_df[
            drivers_df["name_acronym"] == driver_abbr
        ]
        return int(row["driver_number"].iloc[0]) \
               if not row.empty else None

    # =====================================================
    # LATEST LAP DATA
    # =====================================================
    latest = df.sort_values("LapNumber").groupby("Driver").tail(1)

    # =====================================================
    # TOP METRICS
    # =====================================================
    col1, col2, col3, col4 = st.columns(4)

    # Get final race leader from position data
    pos_df = get_positions(session_key)
    if not pos_df.empty and "position" in pos_df.columns:
        last_pos = pos_df.sort_values("date").groupby(
            "driver_number").last().reset_index()
        leader_row = last_pos.sort_values("position").iloc[0]
        leader_num = str(leader_row["driver_number"])
        leader_row2 = drivers_df[
            drivers_df["driver_number"].astype(str) == leader_num
        ]
        leader = leader_row2["name_acronym"].iloc[0] \
                 if not leader_row2.empty else "N/A"
    else:
        leader = latest.sort_values(
            "OvertakeProbability", ascending=False
        ).iloc[0].get("Driver", "N/A")

    highest_prob = round(latest["OvertakeProbability"].max(), 1)
    avg_prob     = round(latest["OvertakeProbability"].mean(), 1)
    total_laps   = int(df["LapNumber"].max())

    with col1: st.metric("🏁 Race Winner", leader)
    with col2: st.metric("🔥 Highest Overtake %", f"{highest_prob}%")
    with col3: st.metric("📊 Avg Overtake %", f"{avg_prob}%")
    with col4: st.metric("🛞 Total Laps", total_laps)

    st.markdown("---")

    # =====================================================
    # OVERTAKE PROBABILITY TABLE
    # =====================================================
    st.subheader("📊 OVERTAKE PROBABILITIES")
    table_cols = [c for c in
                  ["Driver", "TeamName", "Compound",
                   "TyreLife", "OvertakeProbability"]
                  if c in latest.columns]
    table = latest[table_cols].sort_values(
        "OvertakeProbability", ascending=False
    )
    st.dataframe(table, use_container_width=True)

    # =====================================================
    # OVERTAKE BAR CHART
    # =====================================================
    st.subheader("🔥 OVERTAKE INTENSITY")
    bar_fig = px.bar(
        table, x="Driver", y="OvertakeProbability",
        color="OvertakeProbability", template="plotly_dark",
        text_auto=".1f"
    )
    bar_fig.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        font_color="white"
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # =====================================================
    # POSITION CHANGES (from /position endpoint)
    # =====================================================
    st.subheader("📈 DRIVER POSITION CHANGES")
    if not pos_df.empty and "position" in pos_df.columns:
        pos_merged = pos_df.merge(
            drivers_df[["driver_number", "name_acronym"]].rename(
                columns={"driver_number": "driver_number",
                         "name_acronym": "Driver"}
            ),
            on="driver_number", how="left"
        )
        pos_merged["date"] = pd.to_datetime(
            pos_merged["date"], errors="coerce"
        )
        pos_fig = px.line(
            pos_merged.sort_values("date"),
            x="date", y="position", color="Driver",
            template="plotly_dark",
            labels={"date": "Race Time", "position": "Position"}
        )
        pos_fig.update_yaxes(autorange="reversed")
        pos_fig.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            font_color="white"
        )
        st.plotly_chart(pos_fig, use_container_width=True)
    else:
        # Fallback: use lap number from laps data
        if "LapNumber" in df.columns:
            pos_fig = px.line(
                df, x="LapNumber", y="LapNumber",
                color="Driver", template="plotly_dark"
            )
            st.plotly_chart(pos_fig, use_container_width=True)
        else:
            st.info("Position data not available for this race.")

    # =====================================================
    # TYRE DEGRADATION
    # =====================================================
    st.subheader("🛞 TYRE DEGRADATION ANALYSIS")
    if "TyreLife" in df.columns and "LapTimeSeconds" in df.columns:
        tyre_fig = px.scatter(
            df, x="TyreLife", y="LapTimeSeconds",
            color="Compound", template="plotly_dark",
            hover_data=["Driver"] if "Driver" in df.columns else None
        )
        tyre_fig.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            font_color="white"
        )
        st.plotly_chart(tyre_fig, use_container_width=True)
    else:
        st.info("Tyre data not available for this race.")

    # =====================================================
    # PIT STOP STRATEGY
    # =====================================================
    st.subheader("🛞 PIT STOP STRATEGY")
    if not pits_df.empty:
        pit_merged = pits_df.merge(
            drivers_df[["driver_number", "name_acronym"]].rename(
                columns={"name_acronym": "Driver"}
            ),
            on="driver_number", how="left"
        )
        pit_fig = px.scatter(
            pit_merged,
            x="lap_number" if "lap_number" in pit_merged.columns else pit_merged.index,
            y="Driver",
            color="Driver",
            template="plotly_dark",
            title="Pit Stop Timeline"
        )
        pit_fig.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            font_color="white"
        )
        st.plotly_chart(pit_fig, use_container_width=True)
    else:
        st.info("Pit stop data not available for this race.")

    # =====================================================
    # SECTOR PERFORMANCE
    # =====================================================
    st.subheader("⚡ SECTOR PERFORMANCE")
    sector_cols = [c for c in ["Sector1", "Sector2", "Sector3"]
                   if c in df.columns]
    if sector_cols and "Driver" in df.columns:
        sector_avg = df.groupby("Driver")[sector_cols].mean().reset_index()
        sec_fig = go.Figure()
        for s in sector_cols:
            sec_fig.add_trace(
                go.Bar(x=sector_avg["Driver"], y=sector_avg[s], name=s)
            )
        sec_fig.update_layout(
            barmode="group", template="plotly_dark",
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            font_color="white", title="Average Sector Times (s)"
        )
        st.plotly_chart(sec_fig, use_container_width=True)
    else:
        st.info("Sector data not available for this race.")

    # =====================================================
    # FASTEST LAPS
    # =====================================================
    st.subheader("⚡ FASTEST LAPS")
    if "Driver" in df.columns and "LapTimeSeconds" in df.columns:
        fastest = (
            df.groupby("Driver")["LapTimeSeconds"]
            .min().reset_index()
            .sort_values("LapTimeSeconds")
        )
        fast_fig = px.bar(
            fastest, x="Driver", y="LapTimeSeconds",
            color="LapTimeSeconds", template="plotly_dark"
        )
        fast_fig.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            font_color="white"
        )
        st.plotly_chart(fast_fig, use_container_width=True)

    # =====================================================
    # TELEMETRY — SINGLE DRIVER
    # =====================================================
    st.subheader(f"🏎️ SPEED TELEMETRY — {driver_choice}")
    drv_num = get_driver_number(driver_choice)
    if drv_num:
        with st.spinner(f"Loading telemetry for {driver_choice}..."):
            tel_df = get_telemetry(session_key, drv_num)
        if not tel_df.empty and "speed" in tel_df.columns:
            tele_fig = go.Figure()
            tele_fig.add_trace(go.Scatter(
                x=tel_df.index, y=tel_df["speed"],
                mode="lines", name="Speed",
                line=dict(color="#ff1e00")
            ))
            tele_fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0a0a",
                plot_bgcolor="#0a0a0a", font_color="white",
                title=f"{driver_choice} — Speed Trace",
                xaxis_title="Sample", yaxis_title="Speed (km/h)"
            )
            st.plotly_chart(tele_fig, use_container_width=True)
        else:
            st.info(f"Telemetry not available for {driver_choice}.")
    else:
        st.info(f"Could not find driver number for {driver_choice}.")

    st.markdown("---")

    # =====================================================
    # DRIVER vs DRIVER
    # =====================================================
    st.subheader("⚔️ DRIVER vs DRIVER COMPARISON")
    cmp1, cmp2 = st.columns(2)
    with cmp1:
        driver1 = st.selectbox(
            "Driver 1", all_drivers, index=0, key="d1"
        )
    with cmp2:
        driver2 = st.selectbox(
            "Driver 2", all_drivers,
            index=min(1, len(all_drivers)-1), key="d2"
        )

    if driver1 == driver2:
        st.warning("Please select two different drivers.")
    else:
        d1_df = df[df["Driver"] == driver1]
        d2_df = df[df["Driver"] == driver2]

        # Lap times
        st.markdown("#### 📉 Lap Time Battle")
        lt_fig = go.Figure()
        lt_fig.add_trace(go.Scatter(
            x=d1_df["LapNumber"], y=d1_df["LapTimeSeconds"],
            mode="lines+markers", name=driver1,
            line=dict(color="#ff1e00")
        ))
        lt_fig.add_trace(go.Scatter(
            x=d2_df["LapNumber"], y=d2_df["LapTimeSeconds"],
            mode="lines+markers", name=driver2,
            line=dict(color="#00d2ff")
        ))
        lt_fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a", font_color="white",
            title=f"{driver1} vs {driver2} — Lap Times",
            xaxis_title="Lap", yaxis_title="Lap Time (s)"
        )
        st.plotly_chart(lt_fig, use_container_width=True)

        # Sector times
        if sector_cols:
            st.markdown("#### ⚡ Average Sector Times")
            sec_cmp = go.Figure()
            for drv, drv_df, color in [
                (driver1, d1_df, "#ff1e00"),
                (driver2, d2_df, "#00d2ff")
            ]:
                sec_cmp.add_trace(go.Bar(
                    x=sector_cols,
                    y=[drv_df[s].replace(0, np.nan).mean()
                       for s in sector_cols],
                    name=drv, marker_color=color
                ))
            sec_cmp.update_layout(
                barmode="group", template="plotly_dark",
                paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
                font_color="white",
                title=f"{driver1} vs {driver2} — Avg Sector Times"
            )
            st.plotly_chart(sec_cmp, use_container_width=True)

        # Telemetry overlay
        st.markdown("#### 🏎️ Speed Trace Overlay")
        d1_num = get_driver_number(driver1)
        d2_num = get_driver_number(driver2)
        if d1_num and d2_num:
            with st.spinner("Loading telemetry..."):
                tel1 = get_telemetry(session_key, d1_num)
                tel2 = get_telemetry(session_key, d2_num)
            if (not tel1.empty and not tel2.empty and
                    "speed" in tel1.columns and "speed" in tel2.columns):
                ov = go.Figure()
                ov.add_trace(go.Scatter(
                    x=tel1.index, y=tel1["speed"],
                    mode="lines", name=driver1,
                    line=dict(color="#ff1e00")
                ))
                ov.add_trace(go.Scatter(
                    x=tel2.index, y=tel2["speed"],
                    mode="lines", name=driver2,
                    line=dict(color="#00d2ff")
                ))
                ov.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a0a0a",
                    plot_bgcolor="#0a0a0a", font_color="white",
                    title=f"{driver1} vs {driver2} — Speed Trace"
                )
                st.plotly_chart(ov, use_container_width=True)
            else:
                st.info("Telemetry overlay not available.")

        # Head to head summary cards
        st.markdown("#### 📋 Race Summary")
        hc1, hc2 = st.columns(2)
        for col, drv, drv_df, color in [
            (hc1, driver1, d1_df, "#ff1e00"),
            (hc2, driver2, d2_df, "#00d2ff")
        ]:
            best = drv_df["LapTimeSeconds"].replace(0, np.nan).min()
            avg  = drv_df["LapTimeSeconds"].replace(0, np.nan).mean()
            cmpd = ", ".join(
                drv_df["Compound"].dropna().unique()
            ) if "Compound" in drv_df.columns else "N/A"
            with col:
                st.markdown(f"""
                <div style="background:#151515;padding:16px;
                            border-radius:12px;
                            border-left:5px solid {color};">
                    <h3 style="color:white;">{drv}</h3>
                    <p style="color:#ccc;">Best Lap:
                        <b>{round(best,3) if not np.isnan(best) else 'N/A'}s</b>
                    </p>
                    <p style="color:#ccc;">Avg Lap:
                        <b>{round(avg,3) if not np.isnan(avg) else 'N/A'}s</b>
                    </p>
                    <p style="color:#ccc;">Compounds: <b>{cmpd}</b></p>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # =====================================================
    # WEATHER
    # =====================================================
    st.subheader("🌦️ TRACK CONDITIONS")
    if not weather_df.empty:
        wf = go.Figure()
        if "air_temperature" in weather_df.columns:
            wf.add_trace(go.Scatter(
                x=weather_df["date"], y=weather_df["air_temperature"],
                mode="lines", name="Air Temp"
            ))
        if "track_temperature" in weather_df.columns:
            wf.add_trace(go.Scatter(
                x=weather_df["date"], y=weather_df["track_temperature"],
                mode="lines", name="Track Temp"
            ))
        wf.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a", font_color="white",
            title="Temperature During Race"
        )
        st.plotly_chart(wf, use_container_width=True)
    else:
        st.info("Weather data not available for this race.")

    # =====================================================
    # RACE LEADERBOARD
    # =====================================================
    st.subheader("🏆 RACE LEADERBOARD")
    for _, row in latest.sort_values(
        "OvertakeProbability", ascending=False
    ).iterrows():
        drv  = row.get("Driver", "N/A")
        cmpd = row.get("Compound", "N/A")
        life = int(row.get("TyreLife", 0))
        prob = round(row.get("OvertakeProbability", 0), 1)
        st.markdown(f"""
        <div style="background:#151515;padding:10px;border-radius:10px;
                    border-left:5px solid #ff1e00;margin-bottom:10px;">
            <h4 style="color:white;">{drv}</h4>
            <p style="color:#bbb;">
                Tyre: {cmpd} | Tyre Life: {life} laps
                | Overtake Probability: {prob}%
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<center><h4 style='color:#ff1e00'>"
        "Formula 1 Analytics Dashboard</h4></center>",
        unsafe_allow_html=True
    )

with tab_predict:
    render_prediction_tab()
