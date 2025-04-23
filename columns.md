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