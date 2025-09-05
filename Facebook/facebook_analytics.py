# fb_descriptive_charts_all_at_once.py
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt

DB_PARAMS = {
    "dbname": "test1",
    "user": "postgres",
    "password": "admin",
    "host": "localhost",
    "port": "5432",
}
TABLE = "facebook_data_set"
MIN_REACH_FOR_TOPLIST = 1000

def get_conn():
    return psycopg2.connect(**DB_PARAMS)

def fetch_data(date_from: str = None, date_to: str = None) -> pd.DataFrame:
    where = []
    params = {}
    if date_from:
        where.append("publish_time >= %(date_from)s")
        params["date_from"] = date_from
    if date_to:
        where.append("publish_time < %(date_to)s")
        params["date_to"] = date_to
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT
            post_id, page_id, page_name, title, description, post_type,
            duration_sec, publish_time, year, month, day, time, permalink,
            is_crosspost, is_share, funded_content_status,
            reach, shares, comments, reactions,
            seconds_viewed, average_seconds_viewed, impressions
        FROM {TABLE}
        {where_sql};
    """
    with get_conn() as conn:
        df = pd.read_sql(sql, conn, params=params)
    return df

def prep(df: pd.DataFrame) -> pd.DataFrame:
    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

    for c in [
        "duration_sec","reach","shares","comments","reactions",
        "seconds_viewed","average_seconds_viewed","impressions"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["reactions","comments","shares","reach","duration_sec","average_seconds_viewed"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["engagement"] = df[["reactions","comments","shares"]].sum(axis=1, skipna=True)
    df["er"] = df["engagement"] / df["reach"].replace(0, pd.NA)
    df["retention"] = df["average_seconds_viewed"] / df["duration_sec"].replace(0, pd.NA)

    if "publish_time" in df.columns:
        df["hour"] = df["publish_time"].dt.hour
        df["dow"] = df["publish_time"].dt.day_name()
        df["week"] = df["publish_time"].dt.to_period("W").apply(lambda r: r.start_time)
    else:
        df["hour"] = pd.NA; df["dow"] = pd.NA; df["week"] = pd.NA
    return df

def order_weekdays(frame: pd.DataFrame, col="dow") -> pd.DataFrame:
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    if col in frame.columns:
        frame[col] = pd.Categorical(frame[col], categories=order, ordered=True)
    return frame

# ----- Plotters (return list of figures) -----
def plot_overall_kpis(df: pd.DataFrame):
    figs = []
    total_reach = df["reach"].sum(min_count=1)
    total_eng = df["engagement"].sum(min_count=1)
    overall_er = (total_eng / total_reach) if pd.notna(total_reach) and total_reach else None

    fig = plt.figure(figsize=(7,4))
    labels = ["Total Reach","Total Engagement"]
    values = [total_reach or 0, total_eng or 0]
    plt.bar(labels, values)
    plt.title(f"Overall KPIs (ER: {overall_er*100:.2f}% )" if overall_er is not None else "Overall KPIs")
    plt.ylabel("Count")
    plt.tight_layout()
    figs.append(fig)
    return figs

def plot_by_post_type(df: pd.DataFrame):
    figs = []
    if "post_type" not in df.columns:
        return figs

    g = (
        df.groupby("post_type", dropna=False)
          .agg(reach=("reach","sum"), engagement=("engagement","sum"))
          .reset_index()
          .sort_values("reach", ascending=False)
    )
    fig = plt.figure(figsize=(9,5))
    x = range(len(g))
    plt.bar(x, g["reach"], label="Reach")
    plt.bar(x, g["engagement"], alpha=0.5, label="Engagement")
    plt.xticks(x, g["post_type"], rotation=20, ha="right")
    plt.title("Reach & Engagement by Post Type")
    plt.legend()
    plt.tight_layout()
    figs.append(fig)

    g2 = (
        df.groupby("post_type", dropna=False)
          .agg(er=("er","mean"), retention=("retention","mean"))
          .reset_index()
          .sort_values("er", ascending=False)
    )
    fig2 = plt.figure(figsize=(9,5))
    x2 = range(len(g2))
    plt.bar(x2, g2["er"]*100)
    plt.xticks(x2, g2["post_type"], rotation=20, ha="right")
    plt.ylabel("Engagement Rate (%)")
    plt.title("Avg Engagement Rate by Post Type")
    plt.tight_layout()
    figs.append(fig2)
    return figs

def plot_by_dow(df: pd.DataFrame):
    figs = []
    if "dow" not in df.columns:
        return figs
    d = order_weekdays(df.copy())
    g = (
        d.groupby("dow", dropna=False)
         .agg(er=("er","mean"))
         .reset_index()
         .sort_values("dow")
    )
    fig = plt.figure(figsize=(9,5))
    x = range(len(g))
    plt.bar(x, g["er"]*100)
    plt.xticks(x, g["dow"])
    plt.ylabel("Engagement Rate (%)")
    plt.title("Avg Engagement Rate by Day of Week")
    plt.tight_layout()
    figs.append(fig)
    return figs

def plot_by_hour(df: pd.DataFrame):
    figs = []
    if "hour" not in df.columns:
        return figs
    g = (
        df.groupby("hour", dropna=False)
          .agg(er=("er","mean"))
          .reset_index()
          .sort_values("hour")
    )
    fig = plt.figure(figsize=(9,5))
    plt.plot(g["hour"], g["er"]*100, marker="o")
    plt.xticks(g["hour"])
    plt.xlabel("Hour of Day")
    plt.ylabel("Engagement Rate (%)")
    plt.title("Avg Engagement Rate by Hour")
    plt.tight_layout()
    figs.append(fig)
    return figs

def plot_weekly_trends(df: pd.DataFrame):
    figs = []
    if "week" not in df.columns:
        return figs
    g = (
        df.groupby("week", dropna=False)
          .agg(reach=("reach","sum"), engagement=("engagement","sum"))
          .reset_index()
          .sort_values("week")
    )
    fig = plt.figure(figsize=(10,5))
    plt.plot(g["week"], g["reach"], marker="o", label="Reach")
    plt.plot(g["week"], g["engagement"], marker="o", label="Engagement")
    plt.xlabel("Week")
    plt.ylabel("Count")
    plt.title("Weekly Trends: Reach & Engagement")
    plt.legend()
    plt.tight_layout()
    figs.append(fig)
    return figs

def plot_top_bottom(df: pd.DataFrame, min_reach=MIN_REACH_FOR_TOPLIST):
    figs = []
    d = df.copy()
    d = d[d["reach"].fillna(0) >= min_reach]
    d = d[~d["er"].isna()]
    if d.empty:
        return figs

    for c in ["post_id","title","permalink","er","reach","engagement"]:
        if c not in d.columns:
            d[c] = pd.NA
    d["label"] = d["title"].fillna("").apply(lambda s: (s[:40] + "…") if len(s) > 43 else s)

    top10 = d.sort_values("er", ascending=False).head(10)
    bot10 = d.sort_values("er", ascending=True).head(10)

    fig1 = plt.figure(figsize=(10,6))
    plt.barh(range(len(top10)), (top10["er"]*100)[::-1])
    plt.yticks(range(len(top10)), top10["label"][::-1])
    plt.xlabel("Engagement Rate (%)")
    plt.title(f"Top 10 Posts by ER (reach ≥ {min_reach})")
    plt.tight_layout()
    figs.append(fig1)

    fig2 = plt.figure(figsize=(10,6))
    plt.barh(range(len(bot10)), (bot10["er"]*100)[::-1])
    plt.yticks(range(len(bot10)), bot10["label"][::-1])
    plt.xlabel("Engagement Rate (%)")
    plt.title(f"Bottom 10 Posts by ER (reach ≥ {min_reach})")
    plt.tight_layout()
    figs.append(fig2)

    return figs

# ----- MAIN -----
if __name__ == "__main__":
    df = fetch_data()
    if df.empty:
        print("No rows found.")
        raise SystemExit

    df = prep(df)

    figs = []
    figs += plot_overall_kpis(df)
    figs += plot_by_post_type(df)
    figs += plot_by_dow(df)
    figs += plot_by_hour(df)
    figs += plot_weekly_trends(df)
    figs += plot_top_bottom(df)

    # Show everything at once
    plt.show()
