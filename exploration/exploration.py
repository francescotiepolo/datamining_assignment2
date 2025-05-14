#!/usr/bin/python3
import os
import pandas as pd
import matplotlib.pyplot as plt

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df = pd.read_csv(dir + "data/training_set_VU_DM.csv")

def new_customers(df: pd.DataFrame):
    print("new customer count ", df["visitor_hist_starrating"].isnull().sum())

def scatter(df: pd.DataFrame):
    df["booking_bool"] = df["booking_bool"].map(lambda x: True if x == 1 else False)

    booked = df[df["booking_bool"]]
    booked = booked[["srch_id", "prop_location_score1", "prop_location_score2", "price_usd"]]
    booked = booked.rename(columns={"prop_location_score1": "prop_location_score1_selected", "prop_location_score2": "prop_location_score2_selected", "price_usd": "price_usd_selected"})

    df["row_num_in_group"] = df.groupby("srch_id").cumcount()
    row_nums = df.groupby("srch_id").apply(lambda g: g["booking_bool"].idxmax() - g.index[0])
    row_nums = row_nums.reset_index()
    booked["booked_index"] = row_nums[row_nums["srch_id"].isin(booked["srch_id"])][0].to_numpy()
    not_booked = df[df["srch_id"].isin(booked["srch_id"]) & ~df["booking_bool"]]
    del df
    not_booked = not_booked[["srch_id", "prop_location_score1", "prop_location_score2", "price_usd"]]
    booked["prop_location_score1_other"] = not_booked.groupby("srch_id")["prop_location_score1"].mean().tolist()
    booked["price_usd_other"] = not_booked.groupby("srch_id")["price_usd"].mean().tolist()
    figures = dir + "figures"
    if not os.path.isdir(figures):
        os.mkdir(figures)
    plt.hist(booked["booked_index"], bins=list(range(booked["booked_index"].max()+2)))
    plt.savefig(figures + "/place.pdf")
    plt.clf()
    plt.scatter(booked["prop_location_score1_selected"], booked["prop_location_score1_other"])
    plt.savefig(figures + "/location.pdf")
    plt.clf()
    plt.scatter(booked["price_usd_selected"], booked["price_usd_other"])
    plt.savefig(figures + "/price.pdf")

def has_booked(df: pd.DataFrame):
    r = df.groupby("srch_id")["booking_bool"].sum()
    print("booked")
    print(r.sum())
    print("not booked")
    print(r.size - r.sum())
    print(list(r[r == 0].reset_index()["srch_id"]))

scatter(df)
#has_booked(df)