#!/usr/bin/python3
import os
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df = pd.read_csv(dir + "data/training_set_VU_DM.csv")

search_columns = ["srch_id", "date_time", "site_id", "visitor_location_country_id", "visitor_hist_starrating", "visitor_hist_adr_usd", "srch_destination_id", "srch_length_of_stay", "srch_booking_window", "srch_adults_count", "srch_children_count", "srch_room_count", "srch_saturday_night_bool", "srch_query_affinity_score", "orig_destination_distance", "random_bool"]
search_result = ["srch_id", "prop_id", "prop_country_id", "prop_starrating", "prop_review_score", "prop_brand_bool", "prop_location_score1", "prop_location_score2", "prop_log_historical_price", "position", "price_usd", "promotion_flag", "click_bool", "booking_bool"]

#Separate searches from the associated items.
search_df = df[search_columns]
result_df = df[search_result]
#Calculate a rating a user implicitly gives a search result. Click +1, book +2
scoring = pd.DataFrame()
scoring[["srch_id", "prop_id"]] = result_df[["srch_id", "prop_id"]]
scoring["score"] = result_df["click_bool"] + 2*result_df["booking_bool"]

def encode(series, encoder):
    return encoder.fit_transform(series.values.reshape((-1, 1))).astype(int).reshape(-1)

user_encoder, movie_encoder = OrdinalEncoder(), OrdinalEncoder()
scoring["user_id_encoding"] = encode(df["srch_id"], user_encoder)
scoring["movie_id_encoding"] = encode(df["prop_id"], movie_encoder)

X = csr_matrix((scoring["score"], (scoring["user_id_encoding"], scoring["movie_id_encoding"])))
