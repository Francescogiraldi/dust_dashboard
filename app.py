import io
import json
import zipfile
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st
from dateutil import parser as dateparser
import plotly.express as px

st.set_page_config(page_title="Dust Workspace Usage Dashboard", layout="wide")

DEFAULT_WORKSPACE_ID = "08OLJrPMMF"
DUST_SERVERS = {
    "US (us-central1)": "https://dust.tt",
    "EU (europe-west1)": "https://eu.dust.tt",
}

@st.cache_data(show_spinner=False)
def fetch_usage(
    base_url: str,
    workspace_id: str,
    api_key: str,
    start: str,
    mode: str,
    table: str,
    end: Optional[str] = None,
    include_inactive: bool = False,
    output_format: str = "json",
) -> Tuple[requests.Response, str]:
    url = f"{base_url}/api/v1/w/{workspace_id}/workspace-usage"
    params: Dict[str, Any] = {
        "start": start,
        "mode": mode,
        "table": table,
        "includeInactive": str(include_inactive).lower(),
        "format": output_format,
    }
    if mode == "range" and end:
        params["end"] = end

    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, params=params, timeout=60)
    ctype = resp.headers.get("Content-Type", "")
    return resp, ctype


def to_dataframe_from_json_payload(payload: Any) -> Optional[pd.DataFrame]:
    if isinstance(payload, list):
        try:
            return pd.DataFrame(payload)
        except Exception:
            return None
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(v, list):
                try:
                    return pd.DataFrame(v)
                except Exception:
                    continue
        # fallback: if dict of dicts
        try:
            return pd.DataFrame.from_dict(payload, orient="index")
        except Exception:
            return None
    return None


def parse_response(resp: requests.Response, content_type: str) -> Dict[str, Any]:
    if "application/json" in content_type:
        try:
            payload = resp.json()
        except Exception:
            payload = json.loads(resp.text)
        df = to_dataframe_from_json_payload(payload)
        return {"kind": "json", "payload": payload, "df": df}

    if "text/csv" in content_type:
        df = pd.read_csv(io.StringIO(resp.text))
        return {"kind": "csv", "df": df}

    if "application/zip" in content_type:
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        tables: Dict[str, pd.DataFrame] = {}
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                with z.open(name) as f:
                    tables[name] = pd.read_csv(f)
        return {"kind": "zip", "tables": tables}

    return {"kind": "other", "raw": resp.text}


def format_month_value(d: date) -> str:
    return d.strftime("%Y-%m")


def format_day_value(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "created_at",
        "ts",
        "timestamp",
        "time",
        "date",
        "submitted_at",
        "sent_at",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Try to find a column with datetime-like values
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
        if pd.api.types.is_object_dtype(df[c]):
            sample = df[c].dropna().astype(str).head(10)
            try:
                if len(sample) and sum(1 for s in sample if _safe_parse_dt(s)) >= max(3, len(sample) // 2):
                    return c
            except Exception:
                pass
    return None


def _safe_parse_dt(s: str) -> Optional[datetime]:
    try:
        return dateparser.parse(s)
    except Exception:
        return None


def detect_user_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["user", "user_id", "user_email", "email", "customer", "author"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_assistant_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["assistant", "assistant_id", "assistant_name", "agent", "agent_id", "agent_name"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_id_col(df: pd.DataFrame, entity: str) -> Optional[str]:
    if entity == "user":
        for c in ["user_id", "id"]:
            if c in df.columns:
                return c
    if entity == "assistant":
        for c in ["assistant_id", "id"]:
            if c in df.columns:
                return c
    return None


def find_name_col(df: pd.DataFrame, entity: str) -> Optional[str]:
    if entity == "user":
        for c in ["user_name", "name", "user_email", "email"]:
            if c in df.columns:
                return c
    if entity == "assistant":
        for c in ["assistant_name", "name"]:
            if c in df.columns:
                return c
    return None


def build_lookup_map(lookup_df: pd.DataFrame, entity: str) -> Optional[Dict[Any, Any]]:
    id_col = find_id_col(lookup_df, entity)
    name_col = find_name_col(lookup_df, entity)
    if id_col and name_col:
        try:
            return dict(zip(lookup_df[id_col], lookup_df[name_col]))
        except Exception:
            return None
    return None


def ensure_display_columns(df: pd.DataFrame, prefer_names: bool, lookups: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[Optional[str], Optional[str]]:
    user_col = None
    assistant_col = None

    # Prefer existing name columns
    if prefer_names:
        user_col = find_name_col(df, "user")
        assistant_col = find_name_col(df, "assistant")

    # Fallback to IDs
    if user_col is None:
        user_col = find_id_col(df, "user")
    if assistant_col is None:
        assistant_col = find_id_col(df, "assistant")

    # If we still only have IDs and lookups are available, map to names
    if prefer_names and lookups:
        # Assistants
        if assistant_col and "assistants.csv" in lookups and assistant_col.endswith("id") and "assistant_display" not in df.columns:
            amap = build_lookup_map(lookups["assistants.csv"], "assistant")
            if amap:
                try:
                    df["assistant_display"] = df[assistant_col].map(amap).fillna(df[assistant_col].astype(str))
                    assistant_col = "assistant_display"
                except Exception:
                    pass
        # Users
        if user_col and ("users.csv" in lookups) and user_col.endswith("id") and "user_display" not in df.columns:
            umap = build_lookup_map(lookups["users.csv"], "user")
            if umap:
                try:
                    df["user_display"] = df[user_col].map(umap).fillna(df[user_col].astype(str))
                    user_col = "user_display"
                except Exception:
                    pass

    return user_col, assistant_col


def render_insights(df: pd.DataFrame, table: str, prefer_names: bool = True, lookups: Optional[Dict[str, pd.DataFrame]] = None, insight_opts: Optional[Dict[str, bool]] = None):
    st.subheader("Insights")
    if df.empty:
        st.info("Aucune donnée à analyser.")
        return

    time_col = detect_time_column(df)
    user_col, assistant_col = ensure_display_columns(df, prefer_names=prefer_names, lookups=lookups)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lignes", f"{len(df):,}")
    with col2:
        if user_col:
            st.metric("Utilisateurs uniques", f"{df[user_col].nunique():,}")
        else:
            st.metric("Utilisateurs uniques", "N/A")
    with col3:
        if assistant_col:
            st.metric("Assistants uniques", f"{df[assistant_col].nunique():,}")
        else:
            st.metric("Assistants uniques", "N/A")
    with col4:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        st.metric("Colonnes numériques", f"{len(numeric_cols)}")

    charts_container = st.container()

    if insight_opts is None:
        insight_opts = {
            "show_daily_volume": True,
            "show_assistant_over_time": True,
            "show_top_users": True,
            "show_top_assistants": True,
            "show_feedback": True,
        }

    if time_col and insight_opts.get("show_daily_volume", True):
        dt = df[time_col].copy()
        if not pd.api.types.is_datetime64_any_dtype(dt):
            dt = dt.astype(str).map(lambda s: _safe_parse_dt(s))
        ts_df = pd.DataFrame({"date": dt}).dropna()
        ts_df["day"] = ts_df["date"].dt.date
        daily = ts_df.groupby("day").size().reset_index(name="events")
        fig = px.line(daily, x="day", y="events", title="Volumes par jour")
        charts_container.plotly_chart(fig, use_container_width=True)

    if time_col and assistant_col and insight_opts.get("show_assistant_over_time", True):
        df2 = df[[time_col, assistant_col]].copy()
        if not pd.api.types.is_datetime64_any_dtype(df2[time_col]):
            df2[time_col] = df2[time_col].astype(str).map(lambda s: _safe_parse_dt(s))
        df2 = df2.dropna()
        df2["day"] = df2[time_col].dt.date
        grp = df2.groupby(["day", assistant_col]).size().reset_index(name="events")
        fig2 = px.line(grp, x="day", y="events", color=assistant_col, title="Volumes par assistant par jour")
        charts_container.plotly_chart(fig2, use_container_width=True)

    if user_col and insight_opts.get("show_top_users", True):
        top_users = df[user_col].value_counts().reset_index()
        top_users.columns = [user_col, "events"]
        top_users = top_users.head(20)
        fig = px.bar(top_users, x=user_col, y="events", title="Top utilisateurs", height=400)
        charts_container.plotly_chart(fig, use_container_width=True)

    if assistant_col and insight_opts.get("show_top_assistants", True):
        top_as = df[assistant_col].value_counts().reset_index()
        top_as.columns = [assistant_col, "events"]
        top_as = top_as.head(20)
        fig = px.bar(top_as, x=assistant_col, y="events", title="Top assistants", height=400)
        charts_container.plotly_chart(fig, use_container_width=True)

    feedback_cols = [c for c in df.columns if c.lower() in {"rating", "score", "thumbs_up", "thumbs_down", "sentiment"}]
    if feedback_cols and insight_opts.get("show_feedback", True):
        st.write("Feedback (auto-détection)")
        for c in feedback_cols:
            vc = df[c].value_counts(dropna=False).reset_index()
            vc.columns = [c, "count"]
            st.dataframe(vc, use_container_width=True)


def load_demo_data(table: str):
    now = datetime.now()
    days = pd.date_range(end=now.date(), periods=14)
    assistants = ["Agent Alpha", "Agent Beta", "Agent Gamma"]
    users = ["alice@acme.co", "bob@acme.co", "carol@acme.co", "dave@acme.co"]

    if table == "assistant_messages":
        rows = []
        for d in days:
            for a in assistants:
                for u in users[:3]:
                    rows.append({
                        "created_at": d.strftime("%Y-%m-%d"),
                        "assistant_name": a,
                        "user_email": u,
                        "message": "Demo message",
                    })
        return pd.DataFrame(rows)

    if table == "users":
        return pd.DataFrame({
            "user_email": users,
            "messages": [42, 20, 15, 8],
            "activity_level": ["high", "medium", "medium", "low"],
        })

    if table == "builders":
        return pd.DataFrame({
            "builder_email": ["builder1@acme.co", "builder2@acme.co"],
            "messages": [120, 55],
            "activity_level": ["high", "medium"],
        })

    if table == "assistants":
        return pd.DataFrame({
            "assistant_name": assistants,
            "messages": [200, 120, 80],
        })

    if table == "feedback":
        return pd.DataFrame({
            "created_at": [d.strftime("%Y-%m-%d") for d in days[:6]],
            "assistant_name": [assistants[i % 3] for i in range(6)],
            "user_email": [users[i % 4] for i in range(6)],
            "rating": [5, 4, 3, 5, 2, 4],
        })

    if table == "all":
        return {
            "users.csv": load_demo_data("users"),
            "assistant_messages.csv": load_demo_data("assistant_messages"),
            "builders.csv": load_demo_data("builders"),
            "assistants.csv": load_demo_data("assistants"),
            "feedback.csv": load_demo_data("feedback"),
        }

    return pd.DataFrame()


def main():
    st.title("Dust Workspace Usage Dashboard")
    st.caption("Visualisez l’utilisation de votre workspace Dust et les principaux insights.")

    with st.sidebar:
        st.header("Configuration")
        server_label = "US (us-central1)"
        base_url = DUST_SERVERS.get(server_label, list(DUST_SERVERS.values())[0])
        w_id = st.text_input("ID du workspace", value=DEFAULT_WORKSPACE_ID)

        api_key_default = "sk-015efa3661161ddfa14636111843610d"
        if "api_key" not in st.session_state:
            st.session_state["api_key"] = api_key_default or ""
        api_key = st.text_input("API Key (Bearer)", value=st.session_state["api_key"], type="password", help="Vous pouvez aussi définir DUST_API_KEY dans .streamlit/secrets.toml")
        st.session_state["api_key"] = api_key

        mode = st.selectbox("Mode", ["month", "range"], index=0)
        table = st.selectbox(
            "Table",
            ["users", "assistant_messages", "builders", "assistants", "feedback", "all"],
            index=1,
        )
        output_format = st.selectbox("Format", ["json", "csv"], index=0)
        include_inactive = st.checkbox("Inclure inactifs (0 messages)", value=False)
        prefer_names = st.checkbox("Préférer les noms (pas d’ID)", value=True)

        if mode == "month":
            start_month = st.date_input("Mois (YYYY-MM)", value=date.today().replace(day=1))
            start_param = format_month_value(start_month)
            end_param = None
        else:
            start_day = st.date_input("Début (YYYY-MM-DD)", value=date.today().replace(day=1))
            end_day = st.date_input("Fin (YYYY-MM-DD)", value=date.today())
            start_param = format_day_value(start_day)
            end_param = format_day_value(end_day)

        run = st.button("Récupérer les données")

    if "auto_run_once" not in st.session_state:
        st.session_state["auto_run_once"] = False
    should_autorun = bool(api_key) and not st.session_state["auto_run_once"]

    # Demo path
    if False:
        st.session_state["auto_run_once"] = True
        st.info("Mode démo activé: affichage de données fictives.")
        if table != "all":
            df = load_demo_data(table)
            st.subheader("Résultats (DEMO)")
            st.dataframe(df, use_container_width=True)
            render_insights(df, table, prefer_names=prefer_names, insight_opts=insight_opts)
        else:
            tables = load_demo_data("all")
            st.subheader("Résultats (DEMO - ZIP simulé)")
            tab_names = list(tables.keys())
            tabs = st.tabs([n.replace(".csv", "") for n in tab_names])
            for i, (name, df) in enumerate(tables.items()):
                with tabs[i]:
                    st.markdown(f"### {name}")
                    st.dataframe(df, use_container_width=True)
                    render_insights(df, name, prefer_names=prefer_names, lookups=tables, insight_opts=insight_opts)
        st.stop()

    if run or should_autorun:
        st.session_state["auto_run_once"] = True
        if not api_key:
            st.error("Veuillez fournir votre DUST API Key.")
            st.stop()
        with st.spinner("Appel API en cours..."):
            resp, ctype = fetch_usage(
                base_url=base_url,
                workspace_id=w_id,
                api_key=api_key,
                start=start_param,
                end=end_param,
                mode=mode,
                table=table,
                include_inactive=include_inactive,
                output_format=output_format,
            )
        debug_info = {
            "url": f"{base_url}/api/v1/w/{w_id}/workspace-usage",
            "params": {
                "start": start_param,
                "end": end_param,
                "mode": mode,
                "table": table,
                "includeInactive": include_inactive,
                "format": output_format,
            },
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "body_preview": resp.text[:800],
        }
        if resp.status_code == 403:
            st.error("Accès refusé à l’API de données d’usage pour ce workspace (403).")
            st.info("Demande à l’admin d’activer l’accès ‘Workspace Usage API’ pour le workspace 08OLJrPMMF sur https://dust.tt, ou contacte le support.")
            with st.expander("Debug requête API", expanded=False):
                st.write(debug_info)
            st.stop()
        if resp.status_code != 200:
            st.error(f"Erreur API ({resp.status_code}). Détails: {resp.text[:500]}")
            with st.expander("Debug requête API", expanded=False):
                st.write(debug_info)
            st.stop()

        parsed = parse_response(resp, ctype)

        if parsed["kind"] == "json":
            df = parsed.get("df")
            st.subheader("Résultats (JSON)")
            if df is not None and not df.empty:
                # Optional enrichment via remote lookups if only IDs are present
                if prefer_names:
                    try:
                        need_as_map = ("assistant_name" not in df.columns) and ("assistant_id" in df.columns)
                        need_user_map = (not any(c in df.columns for c in ["user_name", "user_email", "email"])) and ("user_id" in df.columns)
                        lookups_remote: Dict[str, pd.DataFrame] = {}
                        if need_as_map:
                            r_as, ct_as = fetch_usage(base_url, w_id, api_key, start_param, mode, "assistants", end=end_param, include_inactive=include_inactive, output_format="json")
                            p_as = parse_response(r_as, ct_as)
                            if p_as.get("df") is not None:
                                lookups_remote["assistants.csv"] = p_as["df"]
                        if need_user_map:
                            r_u, ct_u = fetch_usage(base_url, w_id, api_key, start_param, mode, "users", end=end_param, include_inactive=include_inactive, output_format="json")
                            p_u = parse_response(r_u, ct_u)
                            if p_u.get("df") is not None:
                                lookups_remote["users.csv"] = p_u["df"]
                        if lookups_remote:
                            _ = ensure_display_columns(df, prefer_names=True, lookups=lookups_remote)
                    except Exception:
                        pass
                st.dataframe(df, use_container_width=True)
                render_insights(df, table, prefer_names=prefer_names)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Télécharger CSV", data=csv, file_name=f"{table}.csv", mime="text/csv")
                st.download_button("Télécharger JSON", data=json.dumps(parsed["payload"], ensure_ascii=False).encode("utf-8"), file_name=f"{table}.json", mime="application/json")
            else:
                st.json(parsed["payload"])

        elif parsed["kind"] == "csv":
            df = parsed["df"]
            st.subheader("Résultats (CSV)")
            st.dataframe(df, use_container_width=True)
            render_insights(df, table, prefer_names=prefer_names)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger CSV", data=csv, file_name=f"{table}.csv", mime="text/csv")

        elif parsed["kind"] == "zip":
            st.subheader("Résultats (ZIP de CSVs)")
            tables = parsed["tables"]
            tab_names = list(tables.keys())
            tabs = st.tabs([n.replace(".csv", "") for n in tab_names])
            for i, (name, df) in enumerate(tables.items()):
                with tabs[i]:
                    st.markdown(f"### {name}")
                    st.dataframe(df, use_container_width=True)
                    render_insights(df, name, prefer_names=prefer_names, lookups=tables)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Télécharger CSV", data=csv, file_name=name, mime="text/csv")

        else:
            st.subheader("Résultats (autre)")
            st.code(parsed.get("raw", ""))

    st.sidebar.markdown("---")
    st.sidebar.caption("API protégée par Bearer token. Codes retour: 400/403/404/405 possibles.")


if __name__ == "__main__":
    main()