import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def load_and_prepare_data(filepath):
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(filepath)

    # Create Result column based on the final grade (G3)
    # G3 = 0 is treated as Dropout, not as a zero academic score
    df["Result"] = pd.cut(
        df["G3"],
        bins=[-1, 0, 9, 20],
        labels=["Dropout", "Fail", "Pass"]
    )

    # Calculate percentage out of 100
    df["Percentage"] = (df["G3"] / 20) * 100

    # Calculate the average of weekday and weekend alcohol consumption
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Calculate the average education level of both parents
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Calculate the change in grade from G1 to G3
    df["grade_trend"] = df["G3"] - df["G1"]

    # Count "yes" values across the three support-related columns
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

    # Calculate the average of the first and second period grades
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # Return the complete prepared DataFrame
    return df


def calculate_statistics(df):
    # Select only non-dropout students for academic statistics
    non_dropout_df = df[df["G3"] != 0]

    # Calculate the total number of students
    total_students = len(df)

    # Calculate the mean final grade (G3) excluding dropouts
    class_avg_g3 = np.mean(non_dropout_df["G3"])

    # Calculate the percentage of students who passed (G3 >= 10)
    # Pass rate is calculated only among non-dropout students
    if len(non_dropout_df) > 0:
        pass_rate = (
            np.sum(non_dropout_df["G3"] >= 10)
            / len(non_dropout_df)
        ) * 100
    else:
        pass_rate = 0

    # Count the total number of dropout students
    dropout_count = np.sum(df["G3"] == 0)

    # Count students with G3 between 1 and 9 inclusive
    at_risk_count = np.sum(
        (df["G3"] >= 1) & (df["G3"] <= 9)
    )

    # Create a NumPy correlation matrix for G1, G2, and G3
    # Dropout students are excluded
    correlation_matrix = np.corrcoef(
        non_dropout_df[["G1", "G2", "G3"]].values,
        rowvar=False
    )

    # Return all calculated statistics as a dictionary
    return {
        "total_students": int(total_students),
        "class_avg_g3": class_avg_g3,
        "pass_rate": pass_rate,
        "dropout_count": int(dropout_count),
        "at_risk_count": int(at_risk_count),
        "correlation_matrix": correlation_matrix
    }


def generate_static_charts(df):
    # Create the output folder if it does not already exist
    os.makedirs("output", exist_ok=True)

    # Select non-dropout students for average academic performance
    non_dropout_df = df[df["G3"] != 0]

    # Calculate the average G3 for each studytime level
    studytime_avg = non_dropout_df.groupby("studytime")["G3"].mean()

    # Create the bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(studytime_avg.index, studytime_avg.values)

    # Add title and axis labels
    plt.title("Average G3 by Study Time")
    plt.xlabel("Study Time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)")
    plt.ylabel("Average G3")

    # Set X-axis ticks to studytime levels 1 to 4
    plt.xticks([1, 2, 3, 4])

    # Save the bar chart
    plt.savefig("output/avg_g3_by_studytime.png")

    # Close the chart to free memory
    plt.close()

    # Count the number of students in each result category
    result_counts = df["Result"].value_counts().reindex(
        ["Pass", "Fail", "Dropout"],
        fill_value=0
    )

    # Create the pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(
        result_counts.values,
        labels=result_counts.index,
        autopct="%1.1f%%"
    )

    # Add the chart title
    plt.title("Student Result Distribution")

    # Save the pie chart
    plt.savefig("output/pass_fail_dropout_pie.png")

    # Close the chart to free memory
    plt.close()


def generate_interactive_charts(df):
    # Create an interactive scatter plot showing study time vs final grade
    fig = px.scatter(
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
        title="Study Time vs Final Grade (G3)"
    )

    # Display the interactive scatter plot
    fig.show()

    # Select non-dropout students for average academic performance
    non_dropout_df = df[df["G3"] != 0]

    # Calculate the average G3 for each internet access group
    internet_avg = (
        non_dropout_df
        .groupby("internet", as_index=False)["G3"]
        .mean()
    )

    # Create an interactive bar chart for average G3 by internet access
    fig = px.bar(
        internet_avg,
        x="internet",
        y="G3",
        color="internet",
        title="Average G3 by Internet Access"
    )

    # Display the interactive bar chart
    fig.show()


def print_summary(stats):
    # Print a separator line
    print("=" * 48)

    # Print the system title
    print("STUDENT ACADEMIC RISK INTELLIGENCE SYSTEM")
    print("ANALYSIS SUMMARY")

    # Print another separator line
    print("=" * 48)

    # Print the summary statistics
    print(f"Total Students : {stats['total_students']}")
    print(f"Class Average G3 : {stats['class_avg_g3']:.2f}")
    print(f"Pass Rate : {stats['pass_rate']:.2f}%")
    print(f"At-Risk Count : {stats['at_risk_count']}")
    print(f"Dropout Count : {stats['dropout_count']}")

    # Print the closing separator line
    print("=" * 48)


# Run the complete analysis only when this file is executed directly
if __name__ == "__main__":
    # Load and prepare the student performance dataset
    df = load_and_prepare_data("data/Maths.csv")

    # Calculate the required statistics
    stats = calculate_statistics(df)

    # Generate and save the static Matplotlib charts
    generate_static_charts(df)

    # Generate and display the interactive Plotly charts
    generate_interactive_charts(df)

    # Print the formatted analysis summary
    print_summary(stats)

    # Print the completion message
    print("Analysis complete. Charts saved to output/ folder")