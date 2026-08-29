import streamlit as st
import pandas as pd
import plotly.express as px


# Configure the Streamlit page
st.set_page_config(
    page_title="Student Academic Risk Intelligence System",
    layout="wide",
    page_icon="🎓"
)


# Load and prepare the student performance dataset
@st.cache_data
def load_data():
    # Load Maths.csv from the data folder
    df = pd.read_csv("data/Maths.csv")

    # Create Result column based on the final grade (G3)
    # G3 = 0 is treated as Dropout, not as a zero academic score
    df["Result"] = "Pass"
    df.loc[df["G3"] == 0, "Result"] = "Dropout"
    df.loc[df["G3"].between(1, 9), "Result"] = "Fail"

    # Calculate percentage out of 100
    df["Percentage"] = (df["G3"] / 20) * 100

    # Calculate average weekday and weekend alcohol consumption
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Calculate average parental education level
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Calculate grade trend from G1 to G3
    df["grade_trend"] = df["G3"] - df["G1"]

    # Count "yes" values across support-related columns
    df["total_support"] = (
        (df["schoolsup"] == "yes").astype(int)
        + (df["famsup"] == "yes").astype(int)
        + (df["paid"] == "yes").astype(int)
    )

    # Calculate the academic risk score
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # Calculate the average of G1 and G2
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # Return the prepared DataFrame
    return df


# Load the prepared student data
df = load_data()

# Display the main dashboard title
st.title("🎓 Student Academic Risk Intelligence System")


# Select only non-dropout students for academic performance metrics
non_dropout_df = df[df["G3"] != 0]

# Calculate total number of students
total_students = len(df)

# Check that at least one non-dropout student exists
if len(non_dropout_df) > 0:
    # Calculate average G3 excluding dropout students
    class_average_g3 = round(
        non_dropout_df["G3"].mean(),
        2
    )

    # Calculate pass rate using non-dropout students only
    pass_rate = round(
        (
            (non_dropout_df["G3"] >= 10).sum()
            / len(non_dropout_df)
        ) * 100,
        1
    )
else:
    # Handle the edge case where all students are dropouts
    class_average_g3 = 0.0
    pass_rate = 0.0

# Calculate the number of at-risk students
# At-risk students have G3 between 1 and 9 inclusive
at_risk_count = int(
    df["G3"].between(1, 9).sum()
)


# Create four KPI metric cards in one row
col1, col2, col3, col4 = st.columns(4)

# Display Total Students KPI
with col1:
    st.metric("Total Students", total_students)

# Display Class Average G3 KPI
with col2:
    st.metric("Class Average G3", class_average_g3)

# Display Pass Rate KPI
with col3:
    st.metric("Pass Rate %", f"{pass_rate}%")

# Display At-Risk Count KPI
with col4:
    st.metric("At-Risk Count", at_risk_count)


# Display the Performance Charts section
st.subheader("📊 Performance Charts")

# Create two columns to display charts side by side
left_col, right_col = st.columns(2)


# Create the scatter plot showing study time vs final grade
fig_scatter = px.scatter(
    df,
    x="studytime",
    y="G3",
    color="Result",
    color_discrete_map={
        "Pass": "green",
        "Fail": "red",
        "Dropout": "grey"
    },
    hover_data=["absences", "G1", "G2"],
    title="Study Time vs Final Grade"
)

# Display the scatter plot in the left column
with left_col:
    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )


# Select non-dropout students for academic performance comparison
non_dropout_internet_df = df[df["G3"] != 0]

# Calculate average G3 grouped by internet access
internet_avg = (
    non_dropout_internet_df
    .groupby("internet", as_index=False)["G3"]
    .mean()
)

# Create the bar chart
fig_bar = px.bar(
    internet_avg,
    x="internet",
    y="G3",
    color="internet",
    title="Average G3 by Internet Access"
)

# Display the bar chart in the right column
with right_col:
    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )


# Display the Student Analysis Table section
st.subheader("🚨 Student Analysis Table")

# Create a dropdown to filter students by their result
result_filter = st.selectbox(
    "Filter by Result",
    ["All", "Pass", "Fail", "Dropout"]
)

# Show all students if "All" is selected
if result_filter == "All":
    filtered_df = df
else:
    # Filter students based on the selected result
    filtered_df = df[
        df["Result"] == result_filter
    ]

# Select only the required columns for display
display_columns = [
    "G1",
    "G2",
    "G3",
    "Result",
    "Percentage",
    "absences",
    "studytime",
    "failures",
    "risk_score"
]

# Display the filtered student data
st.dataframe(
    filtered_df[display_columns],
    use_container_width=True
)


# Display the At-Risk Students section
st.subheader("⚠️ At-Risk Students")

# Filter students with G3 between 1 and 9
# and sort them by G3 ascending so the worst-performing students appear first
at_risk_df = (
    df[df["G3"].between(1, 9)]
    .sort_values("G3", ascending=True)
)

# Select only the required columns for the at-risk student table
at_risk_columns = [
    "G1",
    "G2",
    "G3",
    "absences",
    "studytime",
    "failures"
]

# Display the total number of at-risk students
st.write(
    f"Total at-risk students: {len(at_risk_df)}"
)

# Display the at-risk students table
st.dataframe(
    at_risk_df[at_risk_columns],
    use_container_width=True
)