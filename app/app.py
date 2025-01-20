import streamlit as st
import pandas as pd
import pickle
import os



def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)

# Get the absolute path to the models directory
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
kmeans = load_model(os.path.join(models_dir, "kmeans_model.pkl"))
arima = load_model(os.path.join(models_dir, "arima_model.pkl"))

# App title
st.title("Crew Scheduler and Task Allocator")

# Option to upload CSV or enter data manually
st.sidebar.header("Data Input Options")
input_option = st.sidebar.radio("Select input method", ["Upload CSV", "Manual Input"])

# Task details input
st.header("Task Details")
if input_option == "Upload CSV":
    task_file = st.file_uploader("Upload Task Details CSV", type="csv")
    if task_file:
        tasks = pd.read_csv(task_file)
        st.write("Uploaded Task Details:", tasks)
else:
    task_id = st.text_input("Task_ID")
    task_type = st.text_input("Task_Type")
    start_time = st.text_input("Start_Time (HH:MM)")
    end_time = st.text_input("End_Time (HH:MM)")
    task_duration = st.number_input("Task_Duration", min_value=1, step=1)
    if st.button("Add Task"):
        tasks = pd.DataFrame(
            {
                "Task_ID": [task_id],
                "Task_Type": [task_type],
                "Start_Time": [start_time],
                "End_Time": [end_time],
                "Task_Duration ": [task_duration],
            }
        )
        st.write("Task Added:", tasks)
    else:
        tasks = None

# Crew details input
st.header("Crew Details")
if input_option == "Upload CSV":
    crew_file = st.file_uploader("Upload Crew Details CSV", type="csv")
    if crew_file:
        crew = pd.read_csv(crew_file)
        st.write("Uploaded Crew Details:", crew)
else:
    crew_id = st.text_input("Crew ID")
    shift_start = st.text_input("Shift Start_Time (HH:MM)")
    shift_end = st.text_input("Shift End_Time (HH:MM)")
    if st.button("Add Crew Member"):
        crew = pd.DataFrame(
            {
                "Crew ID": [crew_id],
                "Shift Start Time": [shift_start],
                "Shift End Time": [shift_end],
            }
        )
        st.write("Crew Member Added:", crew)
    else:
        crew = None

# Disruption details input
st.header("Disruptions")
if input_option == "Upload CSV":
    disruption_file = st.file_uploader("Upload Disruption Details CSV", type="csv")
    if disruption_file:
        disruptions = pd.read_csv(disruption_file)
        st.write("Uploaded Disruption Details:", disruptions)
else:
    disruption_id = st.text_input("Disruption_ID")
    disruption_type = st.text_input("Type_of_Disruption")
    disruption_time = st.text_input("Timestamp")
    if st.button("Add Disruption"):
        disruptions = pd.DataFrame(
            {
                "Disruption ID": [disruption_id],
                "Disruption Type": [disruption_type],
                "Disruption Time": [disruption_time],
            }
        )
        st.write("Disruption Added:", disruptions)
    else:
        disruptions = None

# Process the data
st.header("Process Data")
if st.button("Run Analysis"):
    if tasks is not None:
        # Task clustering
        task_type_mapping = {task: idx for idx, task in enumerate(tasks["Task_Type"].unique())}
        tasks["Task Type Encoded"] = tasks["Task_Type"].map(task_type_mapping)
        features = tasks[["Task_Type Encoded", "Task_Duration"]]
        tasks["Cluster"] = kmeans.predict(features)
        st.write("Clustered Tasks:", tasks)

    if crew is not None and disruptions is not None:
        st.write("Analysis complete!")
    else:
        st.error("Please provide data for all sections (tasks, crew, disruptions).")

