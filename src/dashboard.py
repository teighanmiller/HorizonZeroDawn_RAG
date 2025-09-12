"""
Contains code for the UI of the Horizon RAG project dashboard.
"""

import streamlit as st
import pandas as pd
import altair as alt


@st.cache_data
def get_data():
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


df_reshaped = get_data()

st.set_page_config(
    page_title="Horizon RAG Chatbot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Dashboard")

alt.theme.enable("dark")

col = st.columns((1.5, 4.5, 2), gap="medium")

with col[0]:

    st.markdown("### User Interaction Metrics")

    ratings = df_reshaped["rating"]

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

    st.metric(
        label="Total Token Count",
        value=aggregate_values(df_reshaped["used_tokens"]),
    )

    st.metric(
        label="Total Number of Queries",
        value=aggregate_values(df_reshaped["query_cnt"]),
    )


with col[1]:
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

    st.subheader("Token by Query Breakdown")

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
    classifications = (
        df_reshaped.groupby("query_classification")
        .size()
        .reset_index(name="Query Count")
        .rename(columns={"query_classification": "classification"})
    )
    st.markdown("#### Top Classifications")

    st.dataframe(
        classifications,
        column_order=("classification", "Query Count"),
        hide_index=True,
        width=None,
        column_config={
            "classification": st.column_config.TextColumn(
                "Classification",
            ),
            "Query Count": st.column_config.ProgressColumn(
                "Count",
                format="%f",
                min_value=0,
                max_value=int(classifications["Query Count"].max()),
            ),
        },
    )
