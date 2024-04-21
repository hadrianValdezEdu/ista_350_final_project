import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import warnings

def filter_string(var):
    var = var.replace(" ", "_")
    var = var.lower()
    return var

def sql_to_dataframe(db_tup):
    return pd.DataFrame(data=db_tup[1], columns=db_tup[0])

def create_national_table(df):
    with sqlite3.connect("safe_to_delete.db") as conn:
        df.to_sql("national_data", con = conn, if_exists= "replace", index= False)

def create_state_table(df):
    with sqlite3.connect("safe_to_delete.db") as conn:
        df.to_sql("state_data", con = conn, if_exists= "replace", index= False)

def run_query(q):
    with sqlite3.connect("safe_to_delete.db") as conn:# create connection
        cur = conn.cursor() # create cursor object

        cur.execute(q) # create execution

        results = cur.fetchall()
        colnames = [n[0] for n in cur.description]

    return (colnames, results)

def run_commit_query(q):
    with sqlite3.connect("safe_to_delete.db") as conn:# create connection
        cur = conn.cursor() # create cursor object
        cur.execute(q) # create execution
        conn.commit()

def clean_df(df):
    def clean_column(var):
        col = df[var]
        col = col.astype(str)
        col = col.str.replace("$", "")
        col = col.str.replace("(", "")
        col = col.str.replace(")", "")
        col = col.str.replace(",", "")
        return col

    df.columns = [filter_string(word) for word in df.columns] # replaces spaces with underscores and makes every letter lowercase
    df = df.loc[:, ['occupation_title_(click_on_the_occupation_title_to_view_its_profile)', 'employment', 'mean_hourly_wage', 'annual_mean_wage', ]] # constrain the df to these columns
    df.columns = ['occupation_title', 'employment', 'mean_hourly_wage', 'annual_mean_wage', ]
    df = df.drop(df.index[0]) # remove the first row

    df = df.reset_index(drop=True)

    df["mean_hourly_wage"] = clean_column("mean_hourly_wage").astype(float)
    df["annual_mean_wage"] = clean_column("annual_mean_wage").astype(float)
    df["employment"] = clean_column("employment").astype(float)

    return df

def make_plot(df, title, xlabel, ylabel):
    df.plot.scatter(x=df.columns[0], y=df.columns[1])
    plt.title(title)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)
    plt.xticks(rotation=45,  ha="right", fontsize=8)
    plt.subplots_adjust(bottom=0.5)

def main():
    national_df = pd.read_html("may_2023_national_occupational_employment_and_wage_estimates_(4_16_2024_8_54_09_am).html")[0]
    state_df = pd.read_html("arizona_may_2023_oews_state_occupational_employment_and_wage_estimates_(4_19_2024_12_21_13_pm).html")[0]
    national_df = clean_df(national_df)
    state_df = clean_df(state_df)

    create_national_table(national_df)
    create_state_table(state_df)

    run_commit_query("DELETE FROM national_data WHERE occupation_title LIKE '%Occupations%'")
    run_commit_query("DELETE FROM state_data WHERE occupation_title LIKE '%Occupations%'")

    national_df = sql_to_dataframe(run_query("SELECT DISTINCT * FROM national_data WHERE mean_hourly_wage > 10 ORDER BY employment DESC LIMIT 20;"))
    state_df = sql_to_dataframe(run_query("SELECT DISTINCT * FROM state_data WHERE mean_hourly_wage > 10 ORDER BY employment DESC LIMIT 20;"))

    make_plot(df=national_df, title="United States of America", xlabel="occupations", ylabel="employment")
    make_plot(df=state_df, title="Arizona", xlabel="occupations", ylabel="employment")
    plt.show()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()