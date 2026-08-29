from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import uvicorn


# Create the FastAPI application
app = FastAPI(
    title="Student Academic Risk Intelligence System API",
    description="API for analyzing student performance data",
    version="1.0.0"
)


def load_data():
    # Load the student performance dataset
    df = pd.read_csv("data/Maths.csv")

    # Create Result column based on the final grade (G3)
    # G3 = 0 is treated as Dropout, not as a zero academic score
    df["Result"] = np.select(
        [
            df["G3"] == 0,
            df["G3"].between(1, 9),
            df["G3"].between(10, 20)
        ],
        [
            "Dropout",
            "Fail",
            "Pass"
        ],
        default="Unknown"
    )

    # Calculate percentage out of 100
    df["Percentage"] = (df["G3"] / 20) * 100

    # Calculate average weekday and weekend alcohol consumption
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Calculate average education level of both parents
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Calculate the grade trend from G1 to G3
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


# Load and prepare the dataset when the application starts
df = load_data()


# Endpoint 1: Return overall academic performance summary
@app.get("/summary")
def get_summary():
    # Select only students who are not dropouts
    non_dropout_df = df[df["G3"] != 0]

    # Calculate total number of students
    total_students = len(df)

    # Check whether there are any non-dropout students
    if len(non_dropout_df) > 0:
        # Calculate average G3 excluding dropout students
        class_average_g3 = round(
            float(non_dropout_df["G3"].mean()),
            2
        )

        # Calculate pass rate among non-dropout students only
        pass_rate_percent = round(
            float(
                (non_dropout_df["G3"] >= 10).mean() * 100
            ),
            2
        )
    else:
        # Handle the edge case where all students are dropouts
        class_average_g3 = 0.0
        pass_rate_percent = 0.0

    # Count students with G3 between 1 and 9 inclusive
    at_risk_count = int(
        df["G3"].between(1, 9).sum()
    )

    # Count dropout students where G3 is 0
    dropout_count = int(
        (df["G3"] == 0).sum()
    )

    # Return the summary as JSON
    return {
        "total_students": total_students,
        "class_average_g3": class_average_g3,
        "pass_rate_percent": pass_rate_percent,
        "at_risk_count": at_risk_count,
        "dropout_count": dropout_count
    }


# Endpoint 2: Return all at-risk students, sorted by lowest G3 first
@app.get("/at-risk")
def get_at_risk_students():
    # Filter students with G3 between 1 and 9
    # and sort by G3 ascending
    at_risk_df = (
        df[df["G3"].between(1, 9)]
        .sort_values("G3", ascending=True)
    )

    # Build and return the required student records
    return [
        {
            "student_index": int(index),
            "G1": int(row["G1"]),
            "G2": int(row["G2"]),
            "G3": int(row["G3"]),
            "absences": int(row["absences"])
        }
        for index, row in at_risk_df.iterrows()
    ]


# Endpoint 3: Return the top 5 non-dropout students by final grade
@app.get("/top-students")
def get_top_students():
    # Exclude dropouts, sort by G3 descending,
    # and select the top 5 students
    top_students_df = (
        df[df["G3"] != 0]
        .sort_values("G3", ascending=False)
        .head(5)
    )

    # Build and return the required student records
    return [
        {
            "student_index": int(index),
            "G1": int(row["G1"]),
            "G2": int(row["G2"]),
            "G3": int(row["G3"])
        }
        for index, row in top_students_df.iterrows()
    ]


# Pydantic model for validating student input data
class StudentInput(BaseModel):
    # G1 must be between 0 and 20
    G1: float = Field(
        ...,
        ge=0,
        le=20,
        description="G1 must be between 0 and 20"
    )

    # G2 must be between 0 and 20
    G2: float = Field(
        ...,
        ge=0,
        le=20,
        description="G2 must be between 0 and 20"
    )

    # Study time level must be between 1 and 4
    studytime: int = Field(
        ...,
        ge=1,
        le=4,
        description="studytime must be between 1 and 4"
    )

    # Absences must be between 0 and 100
    absences: int = Field(
        ...,
        ge=0,
        le=100,
        description="absences must be between 0 and 100"
    )

    # Failures must be between 0 and 4
    failures: int = Field(
        ...,
        ge=0,
        le=4,
        description="failures must be between 0 and 4"
    )


# POST endpoint to predict the student's academic result
@app.post("/predict-result")
def predict_result(student: StudentInput):
    # Calculate the estimated final grade using the given formula
    estimated_g3 = (
        (student.G1 * 0.3)
        + (student.G2 * 0.6)
        + (student.studytime * 0.3)
        - (student.failures * 1.5)
        - (student.absences * 0.05)
    )

    # Clamp the estimated grade between 0 and 20
    estimated_g3 = max(0, min(20, estimated_g3))

    # Round the estimated grade to two decimal places
    estimated_g3 = round(estimated_g3, 2)

    # Determine the predicted academic result
    if estimated_g3 == 0:
        prediction = "Dropout Risk"
    elif estimated_g3 < 10:
        prediction = "Fail"
    else:
        prediction = "Pass"

    # Determine prediction confidence based on G1 and G2
    if student.G1 > 12 and student.G2 > 12:
        confidence = "High"
    elif student.G1 < 8 and student.G2 < 8:
        confidence = "High"
    else:
        confidence = "Medium"

    # Return the prediction results
    return {
        "estimated_g3": estimated_g3,
        "prediction": prediction,
        "confidence": confidence
    }


# Root endpoint: Return basic API information
@app.get("/")
def read_root():
    # Return basic information about the API
    return {
        "message": "Student Academic Risk Intelligence System API",
        "docs": "Visit /docs for full API documentation",
        "version": "1.0.0"
    }


# Run the FastAPI application when this file is executed directly
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )