There are multiple rows per search, each row has the data identifying the search duplicated. These columns are:
* `srch_id`
* `date_time`
* `site_id`
* `visitor_location_country_id`
* `visitor_hist_starrating`
* `visitor_hist_adr_usd`
* `prop_country_id`
* `prop_id`
* `prop_brand_bool`
* `prop_location_score1`
* `prop_location_score2`
* `price_usd`
* `promotion_flag`
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

The other columns are unique per row:
* `prop_country_id`
* `prop_id`
* `prop_starrating`
* `prop_review_score`
* `prop_brand_bool`
* `prop_location_score1`
* `prop_location_score2`
* `prop_log_historical_price`
* `position`
* `price_usd`
* `promotion_flag`
* `gross_booking_usd`
* `compn_rate` (`n` is the value `1` through `8`)
* `compn_inv` (`n` is the value `1` through `8`)
* `compn_rate_percent_diff` (`n` is the value `1` through `8`)

Can be dropped, since it is not available in the test set, and we don't target it
* `position`
* `gross_bookings_usd`

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
* `prop_location_score2` with `null`
    - too much missing and not useful for engineering
* `srch_query_affinity_score` with `null`
    - too much missing and not useful for engineering
* `orig_destination_distance` with `null`
    - too much missing and not useful for engineering

Join?
* `compn_rate` with `null`
* `compn_inv` with `null`
* `compn_rate_percent_diff` with `null`
    - dropped because there existed a difference even when compn_rate = 0 (expedia has same price as competitor)
