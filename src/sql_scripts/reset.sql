-- Drop views
DROP VIEW IF EXISTS single_opponent_view;
DROP VIEW IF EXISTS single_full_view;
DROP VIEW IF EXISTS single_fighter_view;
DROP VIEW IF EXISTS pre_final;

-- Drop tables
DROP TABLE IF EXISTS ufc_fighter_match_stats;
DROP TABLE IF EXISTS ufc_weightclass_stats;
DROP TABLE IF EXISTS ufc_winlossko;
DROP TABLE IF EXISTS weightclass_lookup;

-- Recreate ufc_event_details without 'jevent'
PRAGMA foreign_keys=off;

DROP TABLE IF EXISTS ufc_event_details_new;

CREATE TABLE IF NOT EXISTS ufc_event_details_new AS
SELECT EVENT, URL, DATE, LOCATION  -- include only columns you want to keep
FROM ufc_event_details;

DROP TABLE IF EXISTS ufc_event_details;
ALTER TABLE ufc_event_details_new RENAME TO ufc_event_details;

--ufc_fight_results

DROP TABLE IF EXISTS ufc_fight_results_new;

CREATE TABLE IF NOT EXISTS ufc_fight_results_new AS
SELECT EVENT, BOUT, OUTCOME, WEIGHTCLASS, METHOD, ROUND, [TIME], [TIME FORMAT], REFEREE, DETAILS, URL -- include only columns you want to keep
FROM ufc_fight_results;

DROP TABLE IF EXISTS ufc_fight_results;
ALTER TABLE ufc_fight_results_new RENAME TO ufc_fight_results;

--ufc_fighter_tott

DROP TABLE IF EXISTS ufc_fighter_tott_new;

CREATE TABLE IF NOT EXISTS ufc_fighter_tott_new AS
SELECT FIGHTER, HEIGHT, WEIGHT, REACH, STANCE, DOB, URL -- include only columns you want to keep
FROM ufc_fighter_tott;

DROP TABLE IF EXISTS ufc_fighter_tott;
ALTER TABLE ufc_fighter_tott_new RENAME TO ufc_fighter_tott;

PRAGMA foreign_keys=on;
