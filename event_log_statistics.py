import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import display

CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TS = "time:timestamp"
OUTCOMES = ["A_Pending", "A_Denied", "A_Cancelled"]

def is_categorical(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s))

def log_statistics(logs):
    grouped = logs.groupby(CASE_ID)
    case_length = grouped.size()
    duration_s = (grouped[TS].max() - grouped[TS].min()).dt.total_seconds()

    case_attributes = [c for c in logs.columns if c.startswith("case:")]
    event_attributes = [c for c in logs.columns if c not in case_attributes]
    categorical_event = [c for c in event_attributes if is_categorical(logs[c])]

    return pd.Series({
        "Number of cases": logs[CASE_ID].nunique(),
        "Number of events": len(logs),
        "Number of distinct activities": logs[ACTIVITY].nunique(),
        "Number of variants (activity sequence)": grouped[ACTIVITY].agg("→".join).nunique(),
        "Number of resources": logs["org:resource"].nunique(),
        "Number of distinct case attribute labels": len(case_attributes),
        "Number of distinct event attribute labels": len(event_attributes),
        "Number of categorical event attributes": len(categorical_event),
        "Case length mean (events)": case_length.mean(),
        "Case length std (events)": case_length.std(),
        "Case duration mean (days)": duration_s.mean() / 86400,
        "Case duration std (days)": duration_s.std() / 86400,
        "Case duration mean (minutes)": duration_s.mean() / 60,
        "Case duration std (minutes)": duration_s.std() / 60,
        "Case duration mean (seconds)": duration_s.mean(),
        "Case duration std (seconds)": duration_s.std(),
    })

LAYOUT = [
    ("Log size", "Cases", "Number of cases", None, "{:,.0f}"),
    ("Log size", "Events", "Number of events", None, "{:,.0f}"),
    ("Log size", "Distinct activities", "Number of distinct activities", None, "{:,.0f}"),
    ("Log size", "Resources", "Number of resources", None, "{:,.0f}"),
    ("Variants", "Activity sequence", "Number of variants (activity sequence)", None, "{:,.0f}"),
    ("Attributes", "Case attribute labels", "Number of distinct case attribute labels", None, "{:,.0f}"),
    ("Attributes", "Event attribute labels", "Number of distinct event attribute labels", None, "{:,.0f}"),
    ("Attributes", "Categorical event attributes", "Number of categorical event attributes", None, "{:,.0f}"),
    ("Case length", "Events per case", "Case length mean (events)", "Case length std (events)", "{:,.2f}"),
    ("Case duration", "Days", "Case duration mean (days)", "Case duration std (days)", "{:,.2f}"),
    ("Case duration", "Minutes", "Case duration mean (minutes)", "Case duration std (minutes)", "{:,.0f}"),
    ("Case duration", "Seconds", "Case duration mean (seconds)", "Case duration std (seconds)", "{:,.0f}"),
]
 
 
def pretty_table(stats: dict) -> pd.DataFrame:
    rows, index = [], []
    for section, label, key, std_key, fmt in LAYOUT:
        if std_key:
            label = f"{label} (mean ± std)"
        cells = []
        for s in stats.values():
            v = fmt.format(float(s[key]))
            cells.append(f"{v} ± {fmt.format(float(s[std_key]))}" if std_key else v)
        rows.append(cells)
        index.append((section, label))
    return pd.DataFrame(rows, columns=list(stats),
                        index=pd.MultiIndex.from_tuples(index, names=["", "Statistic"]))
 
 
def style_table(pt: pd.DataFrame):
    return (pt.rename_axis([None, None]).style
            .set_caption("Basic statistics of the BPIC-17 event log")
            .set_table_styles([
                {"selector": "caption", "props": "caption-side: top; font-weight: bold; font-size: 1.05em; padding-bottom: 6px;"},
                {"selector": "th", "props": "text-align: left; padding: 4px 12px;"},
                {"selector": "th.col_heading", "props": "border-bottom: 2px solid #444; text-align: right;"},
                {"selector": "th.level0", "props": "font-weight: bold; vertical-align: top; border-top: 1px solid #bbb;"},
                {"selector": "th.row_heading.level1", "props": "font-weight: normal;"},
                {"selector": "td", "props": "text-align: right; padding: 4px 12px; font-variant-numeric: tabular-nums;"},
                {"selector": "tbody tr:nth-child(even)", "props": "background-color: rgba(127,127,127,0.07);"},
            ]))


def case_outcome(logs):
    last = (logs[logs[ACTIVITY].isin(OUTCOMES)]
            .sort_values(TS).groupby(CASE_ID)[ACTIVITY].last())
    return last.reindex(logs[CASE_ID].unique()).fillna("Not-Completed").rename("outcome")


def case_durations_days(logs):
    g = logs.groupby(CASE_ID)[TS]
    return (g.max() - g.min()).dt.total_seconds() / 86400


def save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    plt.show()

def run(logs, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    logs = logs.sort_values([CASE_ID, TS], kind="stable")

    outcome = case_outcome(logs)
    completed = logs[logs[CASE_ID].isin(outcome[outcome != "Not-Completed"].index)]
    model_log = completed[completed["lifecycle:transition"] == "complete"]

    stats = {"Full log": log_statistics(logs),
             "Completed cases": log_statistics(completed),
             "Modeling log (complete events)": log_statistics(model_log)}
    pt = pretty_table(stats)
    display(style_table(pt))
    pt.to_csv(os.path.join(out_dir, "basic_stats.csv"))
    pd.DataFrame(stats).to_csv(os.path.join(out_dir, "basic_stats_raw.csv"))

    return pt, completed, model_log, outcome