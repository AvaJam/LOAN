import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


# ==========================================
# 1. Load training data
# ==========================================

data = pd.read_csv("data_train.csv")


# ==========================================
# 2. Select features and targets
# ==========================================

features = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_intent",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file"
]

X = data[features]

y_reg = data["loan_amnt"]
y_class = data["loan_status"]


# ==========================================
# 3. Convert categorical variables
# ==========================================

X = pd.get_dummies(
    X,
    columns=[
        "person_gender",
        "person_home_ownership",
        "loan_intent"
    ],
    dtype=int
)


# Education mapping
education_map = {
    "Associate": 4,
    "Bachelor": 1,
    "Doctorate": 3,
    "High School": 0,
    "Master": 2
}

X["person_education"] = X["person_education"].map(
    education_map
)


# Previous loan defaults mapping
previous_loan_defaults_map = {
    "No": 0,
    "Yes": 1
}

X["previous_loan_defaults_on_file"] = (
    X["previous_loan_defaults_on_file"]
    .map(previous_loan_defaults_map)
)


# Save columns used by the models
model_columns = X.columns


# ==========================================
# 4. Train Regression Model
# ==========================================

model_reg = LinearRegression()
model_reg.fit(X, y_reg)


# ==========================================
# 5. Train Classification Model
# ==========================================

model_class = LogisticRegression(
    max_iter=1000
)

model_class.fit(X, y_class)


# ==========================================
# 6. Tkinter Window
# ==========================================

root = tk.Tk()

root.title("LOAN")
root.geometry("500x700")


title = tk.Label(
    root,
    text="Loan Prediction System",
    font=("Arial", 18, "bold")
)

title.pack(pady=15)


# ==========================================
# 7. Input fields
# ==========================================

fields = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_intent",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file"
]

entries = {}


for field in fields:

    row = tk.Frame(root)

    row.pack(
        fill="x",
        padx=10,
        pady=5
    )

    label = tk.Label(
        row,
        text=field,
        width=30,
        anchor="w"
    )

    label.pack(side="left")

    entry = tk.Entry(row)

    entry.pack(
        side="right",
        expand=True,
        fill="x"
    )

    entries[field] = entry


# ==========================================
# 8. Prediction Function
# ==========================================

def predict_loan():

    try:

        # -------------------------------
        # Read user inputs
        # -------------------------------

        input_data = {
            "person_age": float(
                entries["person_age"].get()
            ),

            "person_gender":
                entries["person_gender"].get().strip(),

            "person_education":
                entries["person_education"].get().strip(),

            "person_income": float(
                entries["person_income"].get()
            ),

            "person_emp_exp": float(
                entries["person_emp_exp"].get()
            ),

            "person_home_ownership":
                entries["person_home_ownership"].get().strip(),

            "loan_intent":
                entries["loan_intent"].get().strip(),

            "loan_int_rate": float(
                entries["loan_int_rate"].get()
            ),

            "cb_person_cred_hist_length": float(
                entries["cb_person_cred_hist_length"].get()
            ),

            "credit_score": float(
                entries["credit_score"].get()
            ),

            "previous_loan_defaults_on_file":
                entries[
                    "previous_loan_defaults_on_file"
                ].get().strip()
        }


        # -------------------------------
        # Create DataFrame
        # -------------------------------

        input_df = pd.DataFrame([input_data])


        # -------------------------------
        # Convert categorical variables
        # -------------------------------

        input_df = pd.get_dummies(
            input_df,
            columns=[
                "person_gender",
                "person_home_ownership",
                "loan_intent"
            ],
            dtype=int
        )


        # Education
        input_df["person_education"] = (
            input_df["person_education"]
            .map(education_map)
        )


        # Previous loan defaults
        input_df[
            "previous_loan_defaults_on_file"
        ] = (
            input_df[
                "previous_loan_defaults_on_file"
            ]
            .map(previous_loan_defaults_map)
        )


        # -------------------------------
        # Match training columns
        # -------------------------------

        input_df = input_df.reindex(
            columns=model_columns,
            fill_value=0
        )


        # ==================================
        # Classification
        # ==================================

        class_prediction = model_class.predict(
            input_df
        )[0]


        # ==================================
        # Regression
        # ==================================

        loan_amount = model_reg.predict(
            input_df
        )[0]


        if loan_amount < 0:
            loan_amount = 0


        # ==================================
        # Show result
        # ==================================

        # If the model says loan should NOT
        # be given
        if class_prediction == 0:

            messagebox.showwarning(
                "Loan Decision",
                "Loan should NOT be approved."
            )

        else:

            messagebox.showinfo(
                "Loan Decision",
                "Loan should be approved.\n\n"
                f"Recommended loan amount: "
                f"{loan_amount:,.2f}"
            )


    except ValueError:

        messagebox.showerror(
            "Input Error",
            "Please enter valid values in all fields."
        )


    except Exception as e:

        messagebox.showerror(
            "Error",
            f"An error occurred:\n{e}"
        )


# ==========================================
# 9. Predict Button
# ==========================================

submit_btn = tk.Button(
    root,
    text="PREDICT LOAN",
    command=predict_loan,
    bg="#2ecc71",
    fg="white",
    font=("Arial", 11, "bold"),
    padx=20,
    pady=8
)

submit_btn.pack(pady=20)


# ==========================================
# 10. Start Application
# ==========================================

root.mainloop()

