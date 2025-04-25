

/* Create Win/Loss/KO table to associate which fighter won or lost and if they won by KO */
DROP TABLE IF EXISTS ufc_winlossko;
CREATE TABLE ufc_winlossko AS 
SELECT 
DATE,
trim(ufc_fight_results.EVENT) as EVENT, 
trim(BOUT) as BOUT,
trim(substr(BOUT, 1, instr(BOUT, ' vs. ') - 1)) as fighter, 
CASE WHEN OUTCOME = 'W/L' THEN 1 ELSE 0 END as win,
CASE WHEN OUTCOME = 'L/W' THEN 1 ELSE 0 END as loss,
CASE WHEN METHOD = 'Decision - Unanimous ' AND OUTCOME = 'W/L' THEN 1 ELSE 0 END as udec,
CASE WHEN METHOD = 'Decision - Unanimous ' AND OUTCOME = 'L/W' THEN 1 ELSE 0 END as udecd,
CASE WHEN METHOD = 'Decision - Majority ' AND OUTCOME = 'W/L' THEN 1 ELSE 0 END as mdec,
CASE WHEN METHOD = 'Decision - Majority ' AND OUTCOME = 'L/W' THEN 1 ELSE 0 END as mdecd,
CASE WHEN METHOD = 'Decision - Split ' AND OUTCOME = 'W/L' THEN 1 ELSE 0 END as sdec,
CASE WHEN METHOD = 'Decision - Split ' AND OUTCOME = 'L/W' THEN 1 ELSE 0 END as sdecd,
CASE WHEN METHOD = 'Submission ' AND OUTCOME = 'W/L' THEN 1 ELSE 0 END as subw,
CASE WHEN METHOD = 'Submission ' AND OUTCOME = 'L/W' THEN 1 ELSE 0 END as subwd,
CASE WHEN (METHOD = 'KO/TKO ' OR METHOD = 'TKO - Doctor''s Stoppage ') AND OUTCOME = 'W/L' THEN 1 ELSE 0 END as ko,
CASE WHEN (METHOD = 'KO/TKO ' OR METHOD = 'TKO - Doctor''s Stoppage ') AND OUTCOME = 'L/W' THEN 1 ELSE 0 END as kod,
(((ROUND * 1) - 1) * 300 + CAST(substr(TIME, instr(TIME, ':') + 1) AS INTEGER) + CAST(substr(TIME, 1, instr(TIME, ':') - 1) AS INTEGER) * 60) / 60 as fight_time_minutes
FROM
ufc_fight_results JOIN ufc_event_details 
ON 
trim(ufc_fight_results.EVENT) = trim(ufc_event_details.EVENT)
WHERE NOT(OUTCOME = 'NC/NC'
OR METHOD = 'Overturned')
UNION
SELECT 
DATE,
trim(ufc_fight_results.EVENT), 
trim(BOUT),
trim(substr(BOUT, instr(BOUT, ' vs. ') + 5)) as fighter, 
CASE WHEN OUTCOME = 'L/W' THEN 1 ELSE 0 END as win,
CASE WHEN OUTCOME = 'W/L' THEN 1 ELSE 0 END as loss,
CASE WHEN METHOD = 'Decision - Unanimous ' AND OUTCOME = 'L/W' THEN 1 ELSE 0 END as udec,
CASE WHEN METHOD = 'Decision - Unanimous ' AND OUTCOME = 'W/L' THEN 1 ELSE 0 END as udecd,
CASE WHEN METHOD = 'Decision - Majority ' AND OUTCOME = 'L/W' THEN 1 ELSE 0 END as mdec,
CASE WHEN METHOD = 'Decision - Majority ' AND OUTCOME = 'W/L' THEN 1 ELSE 0 END as mdecd,
CASE WHEN METHOD = 'Decision - Split ' AND OUTCOME = 'L/W' THEN 1 ELSE 0 END as sdec,
CASE WHEN METHOD = 'Decision - Split ' AND OUTCOME = 'W/L' THEN 1 ELSE 0 END as sdecd,
CASE WHEN METHOD = 'Submission ' AND OUTCOME = 'L/W' THEN 1 ELSE 0 END as subw,
CASE WHEN METHOD = 'Submission ' AND OUTCOME = 'W/L' THEN 1 ELSE 0 END as subwd,
CASE WHEN (METHOD = 'KO/TKO ' OR METHOD = 'TKO - Doctor''s Stoppage ') AND OUTCOME = 'L/W' THEN 1 ELSE 0 END as ko,
CASE WHEN (METHOD = 'KO/TKO ' OR METHOD = 'TKO - Doctor''s Stoppage ') AND OUTCOME = 'W/L' THEN 1 ELSE 0 END as kod,
(((ROUND * 1) - 1) * 300 + CAST(substr(TIME, instr(TIME, ':') + 1) AS INTEGER) + CAST(substr(TIME, 1, instr(TIME, ':') - 1) AS INTEGER) * 60) / 60 as fight_time_minutes
FROM
ufc_fight_results JOIN ufc_event_details 
ON 
trim(ufc_fight_results.EVENT) = trim(ufc_event_details.EVENT)
WHERE NOT(OUTCOME = 'NC/NC'
OR METHOD = 'Overturned');


/*
Create match_stats table to collapse ufc_fighter_match_stats which is organized by round
*/
DROP TABLE if exists ufc_fighter_match_stats;
CREATE TABLE ufc_fighter_match_stats AS 
SELECT 
EVENT,
BOUT,
FIGHTER,
SUM(KD) as kd,
SUM(`REV.`) as rev,
SUM(CASE WHEN CTRL IN ('--', '') THEN 0 ELSE CAST(substr(CTRL, 1, instr(CTRL, ':') - 1) AS INTEGER) * 60 + CAST(substr(CTRL, instr(CTRL, ':') + 1) AS INTEGER) END) as ctrl,
SUM(CAST(substr(`SIG.STR.`, 1, instr(`SIG.STR.`, ' of ') - 1) AS INTEGER)) as sigstracc,
SUM(CAST(substr(`SIG.STR.`, instr(`SIG.STR.`, ' of ') + 4) AS INTEGER)) as sigstratt,
SUM(CAST(substr(`TD`, 1, instr(`TD`, ' of ') - 1) AS INTEGER)) as tdacc,
SUM(CAST(substr(`TD`, instr(`TD`, ' of ') + 4) AS INTEGER)) as tdatt,
SUM(`SUB.ATT`) as subatt,
SUM(CAST(substr(`TOTAL STR.`, 1, instr(`TOTAL STR.`, ' of ') - 1) AS INTEGER)) as totalacc,
SUM(CAST(substr(`TOTAL STR.`, instr(`TOTAL STR.`, ' of ') + 4) AS INTEGER)) as totalatt,
SUM(CAST(substr(`HEAD`, 1, instr(`HEAD`, ' of ') - 1) AS INTEGER)) as headacc,
SUM(CAST(substr(`HEAD`, instr(`HEAD`, ' of ') + 4) AS INTEGER)) as headatt,
SUM(CAST(substr(`BODY`, 1, instr(`BODY`, ' of ') - 1) AS INTEGER)) as bodyacc,
SUM(CAST(substr(`BODY`, instr(`BODY`, ' of ') + 4) AS INTEGER)) as bodyatt,
SUM(CAST(substr(`LEG`, 1, instr(`LEG`, ' of ') - 1) AS INTEGER)) as legacc,
SUM(CAST(substr(`LEG`, instr(`LEG`, ' of ') + 4) AS INTEGER)) as legatt,
SUM(CAST(substr(`DISTANCE`, 1, instr(`DISTANCE`, ' of ') - 1) AS INTEGER)) as distacc,
SUM(CAST(substr(`DISTANCE`, instr(`DISTANCE`, ' of ') + 4) AS INTEGER)) as distatt,
SUM(CAST(substr(`CLINCH`, 1, instr(`CLINCH`, ' of ') - 1) AS INTEGER)) as clinchacc,
SUM(CAST(substr(`CLINCH`, instr(`CLINCH`, ' of ') + 4) AS INTEGER)) as clinchatt,
SUM(CAST(substr(`GROUND`, 1, instr(`GROUND`, ' of ') - 1) AS INTEGER)) as groundacc,
SUM(CAST(substr(`GROUND`, instr(`GROUND`, ' of ') + 4) AS INTEGER)) as groundatt
FROM ufc_fight_stats
GROUP BY EVENT, BOUT, FIGHTER;

/*
Some of the CSV files have random spaces or incorrect Bout. Events, and Fighter names, 
- Create a key that removes spaces, and use as a unique record to join.
-- Create Indices for JOINING
*/

-- This is a cleaned up version of Adding jbout, jevent, jfighter along with indexes
-- Adding columns to ufc_fighter_match_stats
ALTER TABLE ufc_fighter_match_stats ADD COLUMN jbout VARCHAR(75);
ALTER TABLE ufc_fighter_match_stats ADD COLUMN jevent VARCHAR(75);
ALTER TABLE ufc_fighter_match_stats ADD COLUMN jfighter VARCHAR(75);
-- Adding indexes to ufc_fighter_match_stats
DROP INDEX IF EXISTS idx_ufc_fighter_match_stats_jbout;
CREATE INDEX idx_ufc_fighter_match_stats_jbout ON ufc_fighter_match_stats(jbout);
DROP INDEX IF EXISTS idx_ufc_fighter_match_stats_jevent;
CREATE INDEX idx_ufc_fighter_match_stats_jevent ON ufc_fighter_match_stats(jevent);
DROP INDEX IF EXISTS idx_ufc_fighter_match_stats_jfighter;
CREATE INDEX idx_ufc_fighter_match_stats_jfighter ON ufc_fighter_match_stats(jfighter);
-- Updating columns in ufc_fighter_match_stats
UPDATE ufc_fighter_match_stats
SET jbout = TRIM(REPLACE(BOUT, ' ', '')),
    jevent = TRIM(REPLACE(EVENT, ' ', '')),
    jfighter = TRIM(REPLACE(FIGHTER, ' ', ''));
-- Adding columns to ufc_fight_results
ALTER TABLE ufc_fight_results ADD COLUMN jbout VARCHAR(75);
ALTER TABLE ufc_fight_results ADD COLUMN jevent VARCHAR(75);
-- Adding indexes to ufc_fight_results
DROP INDEX IF EXISTS idx_ufc_fight_results_jbout;
CREATE INDEX idx_ufc_fight_results_jbout ON ufc_fight_results(jbout);
DROP INDEX IF EXISTS idx_ufc_fight_results_jevent;
CREATE INDEX idx_ufc_fight_results_jevent ON ufc_fight_results(jevent);
-- Updating columns in ufc_fight_results
UPDATE ufc_fight_results
SET jbout = TRIM(REPLACE(BOUT, ' ', '')),
    jevent = TRIM(REPLACE(EVENT, ' ', ''));
-- Adding column to ufc_event_details
ALTER TABLE ufc_event_details ADD COLUMN jevent VARCHAR(75);
-- Adding index to ufc_event_details
DROP INDEX IF EXISTS idx_ufc_event_details_jevent;
CREATE INDEX idx_ufc_event_details_jevent ON ufc_event_details(jevent);
-- Updating column in ufc_event_details
UPDATE ufc_event_details
SET jevent = TRIM(REPLACE(EVENT, ' ', ''));
-- Adding columns to ufc_winlossko
ALTER TABLE ufc_winlossko ADD COLUMN jevent VARCHAR(75);
ALTER TABLE ufc_winlossko ADD COLUMN jbout VARCHAR(75);
ALTER TABLE ufc_winlossko ADD COLUMN jfighter VARCHAR(75);
-- Adding indexes to ufc_winlossko
DROP INDEX IF EXISTS idx_ufc_winlossko_jevent;
CREATE INDEX idx_ufc_winlossko_jevent ON ufc_winlossko(jevent);
DROP INDEX IF EXISTS idx_ufc_winlossko_jbout;
CREATE INDEX idx_ufc_winlossko_jbout ON ufc_winlossko(jbout);
DROP INDEX IF EXISTS idx_ufc_winlossko_jfighter;
CREATE INDEX idx_ufc_winlossko_jfighter ON ufc_winlossko(jfighter);
-- Updating columns in ufc_winlossko
UPDATE ufc_winlossko
SET jevent = TRIM(REPLACE(EVENT, ' ', '')),
    jbout = TRIM(REPLACE(BOUT, ' ', '')),
    jfighter = TRIM(REPLACE(FIGHTER, ' ', ''));
-- Adding column to ufc_fighter_tott
ALTER TABLE ufc_fighter_tott ADD COLUMN jfighter VARCHAR(75);
-- Adding index to ufc_fighter_tott
DROP INDEX IF EXISTS idx_ufc_fighter_tott_jfighter;
CREATE INDEX idx_ufc_fighter_tott_jfighter ON ufc_fighter_tott(jfighter);
-- Updating column in ufc_fighter_tott
UPDATE ufc_fighter_tott
SET jfighter = TRIM(REPLACE(FIGHTER, ' ', ''));
-- Adding composite indexes to ufc_fighter_match_stats
DROP INDEX IF EXISTS idx_ufc_fighter_match_stats_jfighter_jbout;
CREATE INDEX idx_ufc_fighter_match_stats_jfighter_jbout ON ufc_fighter_match_stats(jfighter, jbout);
DROP INDEX IF EXISTS idx_ufc_fighter_match_stats_jevent_jfighter_jbout;
CREATE INDEX idx_ufc_fighter_match_stats_jevent_jfighter_jbout ON ufc_fighter_match_stats(jevent, jfighter, jbout);
-- Adding composite index to ufc_fight_results
DROP INDEX IF EXISTS idx_ufc_fight_results_jevent_jbout;
CREATE INDEX idx_ufc_fight_results_jevent_jbout ON ufc_fight_results(jevent, jbout);
-- Final update for ufc_winlossko
UPDATE ufc_winlossko
SET jfighter = TRIM(REPLACE(FIGHTER, ' ', ''));
