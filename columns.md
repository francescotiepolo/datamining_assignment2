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
* `click_bool`

The columns that can have a value that represents missing data:
* `visitor_hist_starrating` with `null`
* `visitor_hist_adr_usd` with `null`
* `prop_starrating` with `0`
* `prop_review_score` with `0` being no review, and `null` having no data
* `prop_log_historical_price` with `0`
* `prop_location_score2` with `null`
* `srch_query_affinity_score` with `null`
* `orig_destination_distance` with `null`
* `compn_rate` with `null` (`n` is the value `1` through `8`)
* `compn_inv` with `null` (`n` is the value `1` through `8`)
* `compn_rate_percent_diff` with `null` (`n` is the value `1` through `8`)