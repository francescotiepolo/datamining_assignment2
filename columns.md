There are multiple rows per search, each row has the data identifying the search duplicated. These columns are:
* `srch_id`
* `date_time`
* `site_id`
* `visitor_location_country_id`
* `visitor_hist_starrating`
* `visitor_hist_adr_usd`
* `srch_destination_id`
* `srch_length_of_stay`
* `srch_booking_window`
* `srch_adults_count`
* `srch_children_count`
* `srch_room_count`
* `srch_saturday_night_bool`
* `srch_query_affinity_score`
* `orig_destination_distance`
* `random_bool`

The other columns are unique per row

Can be dropped, since it is not available in the test set, and we don't target it
* `position`
* `gross_bookings_usd`
* `click_bool`

The columns that can have a value that represents missing data:
* `visitor_hist_starrating` with `null`
    - can be converted in dummy for existing purchase history or not
* `visitor_hist_adr_usd` with `null`
    - same as `visitor_hist_starrating` (has to check why differenced in % of missing values);
    - probably better than `visitor_hist_starrating` since at time of reservation star rating might have been unavailable, but price always available
* `prop_starrating` with `0`
    - can be filled
* `prop_review_score` with `0` being no review, and `null` having no data
    - can be filled (watch out for the 0: it is NOT missing value!!!)
    - first create a dummy variable that says if the hotel had any reviews and then fill in both the null and the 0 values
* `prop_log_historical_price` with `0`
    - too much missing and not useful for engineering
* `srch_query_affinity_score` with `null`
    - too much missing and not useful for engineering
* `orig_destination_distance` with `null`
    - too much missing and not useful for engineering
* `compn_rate` with `null`
    - too much missing and not useful for engineering
* `compn_inv` with `null`
    - too much missing and not useful for engineering
* `compn_rate_percent_diff` with `null`
    - too much missing and not useful for engineering