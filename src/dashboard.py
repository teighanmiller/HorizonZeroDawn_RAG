"""
Contains code for the UI of the Horizon RAG project dashboard.
"""

import streamlit as st
import pandas as pd
import altair as alt


@st.cache_data(ttl=20)
def get_data():
    """
    Reads the data from the user metrics file into a Pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing all tracked application metrics.
    """
    return pd.read_csv("data/user_data.csv")


def aggregate_values(values: pd.Series) -> int:
    """
    Sums all numbers in a Series.

    Args:
        values (pd.Series): the Series of numbers to sum

    Returns:
        int: the sum of the numbers
    """
    count = values[values.notna()].to_list()
    return sum(count)


def get_unique_cnt(series: pd.Series) -> pd.DataFrame:
    """
    Counts the number of occurances of each unique item in a Series.

    Args:
        classifications (pd.Series): the list of items

    Returns:
        pd.DataFrame: a list of unqiue items with associated counts
    """
    return pd.DataFrame(series.value_counts())


# Load the data
df_reshaped = get_data()

# Set page settings
st.set_page_config(
    page_title="Horizon RAG Chatbot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Dashboard")

alt.theme.enable("dark")

# Set column width and spacing
col = st.columns((1.5, 4.5, 2), gap="large")

with col[0]:
    # Create a graph for likes, dislikes and not rated metrics
    st.markdown("### User Interaction Metrics")

    ratings = df_reshaped["rating"]

    # Calculate rating metrics
    pos = int(aggregate_values(ratings))
    neg = int(
        len(ratings[ratings.notna()]) - aggregate_values(ratings[ratings.notna()])
    )
    nan = int(len(ratings[ratings.isna()]))

    df_tokens = pd.DataFrame(
        {
            "Rating Type": ["Dislikes", "Likes", "Unrated"],
            "Number of Ratings": [neg, pos, nan],
        }
    )

    chart = (
        alt.Chart(df_tokens)
        .mark_bar()  # width=alt.RelativeBandSize(0.9))
        .encode(
            x=alt.X(
                "Rating Type:N",
            ),  # integers only
            y=alt.Y("Number of Ratings:Q"),
            tooltip=["Rating Type", "Number of Ratings"],
        )
        .properties(height=250)
    )

    st.altair_chart(chart, use_container_width=True)

    # Create counters for tokens and number of queries
    st.metric(
        label="Total Token Count",
        value=aggregate_values(df_reshaped["used_tokens"]),
    )

    st.metric(
        label="Total Number of Queries",
        value=aggregate_values(df_reshaped["query_cnt"]),
    )


with col[1]:
    # Graph of the processing time breakdown for RAG per query in the app.
    st.subheader("Processing Time Breakdown for RAG")

    df_plot = df_reshaped[
        [
            "generation_time",
            "rag_time",
            "reword_time",
            "full_response_time",
        ]
    ].copy()

    # Create integer x values
    df_plot["Query"] = range(1, len(df_plot) + 1)

    # Melt the DataFrame so we get "Stage" and "Time" columns
    df_melted = df_plot.melt(
        id_vars=["Query"],
        value_vars=["generation_time", "rag_time", "reword_time", "full_response_time"],
        var_name="Stage",
        value_name="Time",
    )

    chart = (
        alt.Chart(df_melted)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Query:Q", axis=alt.Axis(format="d", tickMinStep=1)
            ),  # integers only
            y=alt.Y("Time:Q", title="Time (s)"),
            color=alt.Color("Stage:N", title="Stage"),
            tooltip=["Query", "Stage", "Time"],
        )
        .properties(height=250)
    )

    st.altair_chart(chart, use_container_width=True)

    # Create a chart for average response time
    st.markdown("#### Average Response Time Trend")

    df_reshaped["timestamp"] = pd.to_datetime(df_reshaped["timestamp"])

    # Group by date and compute average
    avg_response_time = (
        df_reshaped.groupby(df_reshaped["timestamp"].dt.date)["full_response_time"]
        .mean()
        .reset_index()
        .rename(
            columns={"timestamp": "Date", "full_response_time": "Avg Response Time"}
        )
    )

    # Create line chart
    line_chart = (
        alt.Chart(avg_response_time)
        .mark_line(point=True)
        .encode(
            x="Date:T",
            y="Avg Response Time:Q",
            tooltip=["Date:T", "Avg Response Time:Q"],
        )
        .properties(height=300)
    )

    st.altair_chart(line_chart, use_container_width=True)

    # Create a graph for token usage by query.
    st.subheader("Tokens by Query Breakdown")

    df_tokens = pd.DataFrame(
        {
            "Query": range(1, len(df_reshaped) + 1),
            "Tokens": df_reshaped["used_tokens"],
        }
    )

    chart = (
        alt.Chart(df_tokens)
        .mark_bar(width=alt.RelativeBandSize(0.9))
        .encode(
            x=alt.X(
                "Query:Q", axis=alt.Axis(format="d", tickMinStep=1)
            ),  # integers only
            y=alt.Y("Tokens:Q"),
            tooltip=["Query", "Tokens"],
        )
        .properties(height=200)
    )

    st.altair_chart(chart, use_container_width=True)


with col[2]:
    # Group classifications
    classifications = (
        df_reshaped.groupby("query_classification")
        .size()
        .reset_index(name="Query Count")
        .rename(columns={"query_classification": "Classification"})
    )

    st.markdown("#### Top Classifications")

    # Create a pie chart
    pie_chart = (
        alt.Chart(classifications)
        .mark_arc()
        .encode(
            theta=alt.Theta("Query Count:Q", title="Count"),
            color=alt.Color(
                "Classification:N", legend=alt.Legend(title="Classification")
            ),
            tooltip=["Classification", "Query Count"],
        )
    )

    st.altair_chart(pie_chart, use_container_width=True)
