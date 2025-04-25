-- Active: 1715822372322@@127.0.0.1@3306

-- How to run:
-- Put this file in the project base UFC stuff where ./import is a subdirectory
-- terminal
--  sqlite3 ufc.db <UFC-stuff-sqlite3.sql
-- Note: ufc.db and export ufdb.cvs will be created in the ./import directory


-- Specify base directory to use for importing exporting
.cd "/Users/ekellener/Documents/workspace/evan_stuff/UFC stuff/import/"

/*
:Files
Webscraper from UFC website  (scrape_UFC_stats)
Produces 6 csv files:
    scrape_ufc_stats_all_historical_data.ipynb
        ufc_event_details.csv
        ufc_fight_details.csv(may not need)
        ufc_fight_results.csv -> fix weird fight round format, use hand documented wonky_ufc_fight_times file
                              -> get wieghtclass and sex information from ufc_weightclass_stats table
        ufc_fight_stats.csv
    scrape_ufc_stats_fighter_tott.ipynb
        ufc_fighter_details.csv(may not need)
        ufc_fighter_tott.csv -> fillin height, weight, reach in notebook
                             -> come back as filtered_data_final.csv
                             -> export only data with no dob into dob_scrapper.py
                             -> scrape dob data
                             -> import new dob data and combine with height, weight, reach data
                             -> create sex column and remover rows where sex is null
Additional Files Created
    ufc_weightclass_stats -> contains weight class alignmenet with index and sex
                            for example wlterweight would be 09 for index, 170 for weight, 02 for male
    ufc_winlossko -> hold information about who won the fight and the method
    ufc_fighter_match_stats -> contains fight information such as strikes and takedown summaried by fighter and fight
                            -> created additional stripped columns

Steps:
    Clean up & Import key ufc_files
        Formats String to Date formats
    Fill in Gaps on HEIGHT/WEIGHT/REACH and DOB
        Export H/W/R data for Modeling
        Run model to predict HEIGHT/WEIGHT/REACH
        Import predicted values and update
        Export Missing Fighter DOBs
        Run Scraper to find missing DOBs
        Update DOBs
        Remove any Fighters with missing DOBs or H/W/R
    Fill in manually entered "Wonky" Fight round formats (/wonky_ufc_fight_times.csv)
        Remove any results that can't be procssed with the wonky time formats
    Create WEIGHTCLASS index, and sex (1-Female, 2-Male), and assign to Fights
        CREATE WEIGHTCLASS lookup
        ASSIGN to WEIGHTCLASS stats
        Remove any FIGHTERS that can't be assigned a sex
    CREATE a winlossko tracker
    Summarize Match Stats and Add better JOIN keys for Bouts, Fighters, Events
    Create full View that aggregates all features including prepost comp +3, +5 variables
    Create separate Opponent/Fighter Views of the above
    Combine the Opponent/Fighter so the full features are available in the same Fight Row
    Export results.

*/

-- Clean up to remove any dependent views.

DROP VIEW IF EXISTS single_fighter_view;
DROP VIEW IF EXISTS single_opponent_view;
DROP VIEW IF EXISTS single_full_view;
DROP VIEW IF EXISTS pre_final;

-- Import ufc_event_details
DROP TABLE IF EXISTS ufc_event_details;
CREATE TABLE ufc_event_details 
(`EVENT` VARCHAR(75), `URL` VARCHAR(75), `DATE` VARCHAR(75), `LOCATION` VARCHAR(75));
-- Imports CSV directly and creates the table in 1 query using the column names in the CSV
.import --csv --skip 1  'ufc_event_details.csv' ufc_event_details

-- Format the Event Date so it's readable as a date
ALTER TABLE ufc_event_details
ADD COLUMN format_date DATE;

UPDATE ufc_event_details
SET format_date =  
    strftime('%Y-%m-%d', 
        substr(Date, instr(Date, ',') + 2, 4) || '-' ||
        CASE 
            WHEN substr(UPPER(Date), 1, 3) = 'JAN' THEN '01'
            WHEN substr(UPPER(Date), 1, 3) = 'FEB' THEN '02'
            WHEN substr(UPPER(Date), 1, 3) = 'MAR' THEN '03'
            WHEN substr(UPPER(Date), 1, 3) = 'APR' THEN '04'
            WHEN substr(UPPER(Date), 1, 3) = 'MAY' THEN '05'
            WHEN substr(UPPER(Date), 1, 3) = 'JUN' THEN '06'
            WHEN substr(UPPER(Date), 1, 3) = 'JUL' THEN '07'
            WHEN substr(UPPER(Date), 1, 3) = 'AUG' THEN '08'
            WHEN substr(UPPER(Date), 1, 3) = 'SEP' THEN '09'
            WHEN substr(UPPER(Date), 1, 3) = 'OCT' THEN '10'
            WHEN substr(UPPER(Date), 1, 3) = 'NOV' THEN '11'
            WHEN substr(UPPER(Date), 1, 3) = 'DEC' THEN '12'
        END || '-' ||
        CASE 
            WHEN length(trim(substr(Date, instr(Date, ' ') + 1, instr(Date, ',') - instr(Date, ' ') - 1))) = 1 
                THEN '0' || trim(substr(Date, instr(Date, ' ') + 1, instr(Date, ',') - instr(Date, ' ') - 1))
            ELSE 
                trim(substr(Date, instr(Date, ' ') + 1, instr(Date, ',') - instr(Date, ' ') - 1))
        END
    );

select * from ufc_event_details;
ALTER TABLE ufc_event_details
DROP COLUMN `DATE`;
ALTER TABLE ufc_event_details
RENAME COLUMN format_DATE to DATE;


-- Import ufc_fight_details
DROP TABLE IF EXISTS ufc_fight_details;
CREATE TABLE ufc_fight_details (`EVENT` VARCHAR(75), `BOUT` VARCHAR(75), `URL` VARCHAR(75));
.import --csv --skip 1  'ufc_fight_details.csv' ufc_fight_details

-- Import ufc_fight_results
DROP TABLE IF EXISTS ufc_fight_results;
CREATE TABLE ufc_fight_results (`EVENT` VARCHAR(75), `BOUT` VARCHAR(75), `OUTCOME` VARCHAR(75), `WEIGHTCLASS` VARCHAR(75), `METHOD` VARCHAR(75), `ROUND` VARCHAR(75), `TIME` VARCHAR(75), `TIME FORMAT` VARCHAR(75), `REFEREE` VARCHAR(75), `DETAILS` VARCHAR(255), `URL` VARCHAR(75));
.import --csv --skip 1  'ufc_fight_results.csv' ufc_fight_results

-- Import ufc_fight_stats
DROP TABLE IF EXISTS ufc_fight_stats;
CREATE TABLE ufc_fight_stats (`EVENT` VARCHAR(75), `BOUT` VARCHAR(75), `ROUND` VARCHAR(75), `FIGHTER` VARCHAR(75), `KD` VARCHAR(25), `SIG.STR.` VARCHAR(25),`SIG.STR.%` VARCHAR(25), `TOTAL STR.` VARCHAR(25), `TD` VARCHAR(25), `TD%` VARCHAR(25), `SUB.ATT` VARCHAR(25), `REV` VARCHAR(25), `CTRL` VARCHAR(25), `HEAD` VARCHAR(25),`BODY` VARCHAR(25),`LEG` VARCHAR(25), `DISTANCE` VARCHAR(25), `CLINCH` VARCHAR(25), `GROUND` VARCHAR(25));
.import --csv --skip 1  'ufc_fight_stats.csv' ufc_fight_stats

-- Import ufc_fighter_details
DROP TABLE IF EXISTS ufc_fighter_details;
CREATE TABLE ufc_fighter_details (`FIRST` VARCHAR(75), `LAST` VARCHAR(75), `NICKNAME` VARCHAR(75), `URL` VARCHAR(75));
.import --csv --skip 1  'ufc_fighter_details.csv' ufc_fighter_details

-- Import ufc_fighter_tott
DROP TABLE IF EXISTS ufc_fighter_tott;
CREATE TABLE ufc_fighter_tott (`FIGHTER` VARCHAR(75), `HEIGHT` VARCHAR(25), `WEIGHT` VARCHAR(25), `REACH`VARCHAR(25), `STANCE` VARCHAR(75), `DOB` VARCHAR(25), `URL` VARCHAR(75));
.import --csv --skip 1  'ufc_fighter_tott.csv' ufc_fighter_tott

-- Format the DOB Date
ALTER TABLE ufc_fighter_tott
ADD COLUMN `format_DOB` DATE;


UPDATE ufc_fighter_tott
SET format_DOB =  
    strftime('%Y-%m-%d', 
        substr(DOB, instr(DOB, ',') + 2, 4) || '-' ||
        CASE 
            WHEN substr(UPPER(DOB), 1, 3) = 'JAN' THEN '01'
            WHEN substr(UPPER(DOB), 1, 3) = 'FEB' THEN '02'
            WHEN substr(UPPER(DOB), 1, 3) = 'MAR' THEN '03'
            WHEN substr(UPPER(DOB), 1, 3) = 'APR' THEN '04'
            WHEN substr(UPPER(DOB), 1, 3) = 'MAY' THEN '05'
            WHEN substr(UPPER(DOB), 1, 3) = 'JUN' THEN '06'
            WHEN substr(UPPER(DOB), 1, 3) = 'JUL' THEN '07'
            WHEN substr(UPPER(DOB), 1, 3) = 'AUG' THEN '08'
            WHEN substr(UPPER(DOB), 1, 3) = 'SEP' THEN '09'
            WHEN substr(UPPER(DOB), 1, 3) = 'OCT' THEN '10'
            WHEN substr(UPPER(DOB), 1, 3) = 'NOV' THEN '11'
            WHEN substr(UPPER(DOB), 1, 3) = 'DEC' THEN '12'
        END || '-' ||
        CASE 
            WHEN length(trim(substr(DOB, instr(DOB, ' ') + 1, instr(DOB, ',') - instr(DOB, ' ') - 1))) = 1 
                THEN '0' || trim(substr(DOB, instr(DOB, ' ') + 1, instr(DOB, ',') - instr(DOB, ' ') - 1))
            ELSE 
                trim(substr(DOB, instr(DOB, ' ') + 1, instr(DOB, ',') - instr(DOB, ' ') - 1))
        END
    );


ALTER TABLE ufc_fighter_tott 
DROP COLUMN `DOB`;
ALTER TABLE ufc_fighter_tott 
RENAME COLUMN format_DOB to DOB;


-- Format HEIGHT, WEIGHT, REACH columns.
-- Imported file from Height_weight_reach.ipynb that predicts missing values for Reach, Height, and Weight
-- Runs Model Height_weight_reach.ipynb outputs filtered_data_final.csv

DROP TABLE IF EXISTS filtered_height_weight_reach;
CREATE TABLE filtered_height_weight_reach (`FIGHTER` VARCHAR(75), `HEIGHT` FLOAT, `WEIGHT` FLOAT, `REACH` FLOAT, `STANCE` VARCHAR(75), `DOB` VARCHAR(75), `URL` VARCHAR(75));
.import --csv --skip 1  'filtered_data_final.csv' filtered_height_weight_reach


--Create final_data_with_no_dob.csv to fill in DOB's from dob_scrapper.py
-- TODO CREATE sqlite3 export of no_dob.csv

/* Uncomment when hooked to notebook
SELECT 'FIGHTER', 'HEIGHT', 'WEIGHT', 'REACH', 'STANCE', 'DOB', 'URL'
UNION ALL
select * 
INTO OUTFILE '/usr/var/ufc_stats/final_data_with_no_dob.csv'
FIELDS TERMINATED BY ','
    ENCLOSED BY '"'
LINES TERMINATED BY '\n'
from filtered_height_weight_reach
WHERE `DOB` = '--' OR `DOB` = ''
*/
-- Run scraper: dob_scrapper.py to produce final_data_with_dob.csv
-- scrapes sherdog.dog

--Import scraper output to height_weight_reach_dob_complete

DROP TABLE IF EXISTS height_weight_reach_dob_complete;
CREATE TABLE height_weight_reach_dob_complete (`FIGHTER` VARCHAR(75), `HEIGHT` FLOAT, `WEIGHT` FLOAT, `REACH` FLOAT, `STANCE` VARCHAR(75), `DOB` VARCHAR(75), `URL` VARCHAR(75));
.import --csv --skip 1  'final_data_with_dob.csv' height_weight_reach_dob_complete


-- update filtered_height_weight_reach with the completed DOB's from height_weight_reach_dob_complete
UPDATE filtered_height_weight_reach
SET DOB = (SELECT fhwrdc.DOB FROM height_weight_reach_dob_complete fhwrdc WHERE
filtered_height_weight_reach.URL = fhwrdc.URL)
WHERE filtered_height_weight_reach.DOB = '--' or filtered_height_weight_reach.DOB = '';
-- delete rows with DOB = 'Date of birth not found' or 'Fighter not found'
DELETE FROM filtered_height_weight_reach
WHERE DOB in ('Date of birth not found', 'Fighter not found');


-- Adjust DOB to date format (May need not need if it's fixed from the python script)
ALTER TABLE filtered_height_weight_reach
ADD COLUMN format_DOB DATE;

UPDATE filtered_height_weight_reach
SET format_DOB =  
    strftime('%Y-%m-%d', 
        substr(DOB, instr(DOB, ',') + 2, 4) || '-' ||
        CASE 
            WHEN substr(UPPER(DOB), 1, 3) = 'JAN' THEN '01'
            WHEN substr(UPPER(DOB), 1, 3) = 'FEB' THEN '02'
            WHEN substr(UPPER(DOB), 1, 3) = 'MAR' THEN '03'
            WHEN substr(UPPER(DOB), 1, 3) = 'APR' THEN '04'
            WHEN substr(UPPER(DOB), 1, 3) = 'MAY' THEN '05'
            WHEN substr(UPPER(DOB), 1, 3) = 'JUN' THEN '06'
            WHEN substr(UPPER(DOB), 1, 3) = 'JUL' THEN '07'
            WHEN substr(UPPER(DOB), 1, 3) = 'AUG' THEN '08'
            WHEN substr(UPPER(DOB), 1, 3) = 'SEP' THEN '09'
            WHEN substr(UPPER(DOB), 1, 3) = 'OCT' THEN '10'
            WHEN substr(UPPER(DOB), 1, 3) = 'NOV' THEN '11'
            WHEN substr(UPPER(DOB), 1, 3) = 'DEC' THEN '12'
        END || '-' ||
        CASE 
            WHEN length(trim(substr(DOB, instr(DOB, ' ') + 1, instr(DOB, ',') - instr(DOB, ' ') - 1))) = 1 
                THEN '0' || trim(substr(DOB, instr(DOB, ' ') + 1, instr(DOB, ',') - instr(DOB, ' ') - 1))
            ELSE 
                trim(substr(DOB, instr(DOB, ' ') + 1, instr(DOB, ',') - instr(DOB, ' ') - 1))
        END
    );



ALTER TABLE filtered_height_weight_reach DROP COLUMN `DOB`;
ALTER TABLE filtered_height_weight_reach
RENAME COLUMN format_DOB to DOB;





-- Add some indexes to improve performance
DROP INDEX IF EXISTS ft_url;
CREATE INDEX ft_url ON ufc_fighter_tott(URL);
DROP INDEX IF EXISTS fhwr_url;
CREATE INDEX fhwr_url ON filtered_height_weight_reach(URL);

-- Update fighter tott with the revised H, W, R and DOB
UPDATE ufc_fighter_tott
SET 
    HEIGHT = (SELECT HEIGHT FROM filtered_height_weight_reach WHERE filtered_height_weight_reach.URL = ufc_fighter_tott.URL),
    WEIGHT = (SELECT WEIGHT FROM filtered_height_weight_reach WHERE filtered_height_weight_reach.URL = ufc_fighter_tott.URL),
    REACH = (SELECT REACH FROM filtered_height_weight_reach WHERE filtered_height_weight_reach.URL = ufc_fighter_tott.URL),
    DOB = (SELECT DOB FROM filtered_height_weight_reach WHERE filtered_height_weight_reach.URL = ufc_fighter_tott.URL)
WHERE EXISTS (
    SELECT 1 
    FROM filtered_height_weight_reach 
    WHERE filtered_height_weight_reach.URL = ufc_fighter_tott.URL
);


-- deleting rows in tott where DOB is not found
DELETE FROM ufc_fighter_tott
WHERE DOB in ('--', '');

/*
-- DON"T UNCOMMENT WILL OVERWRITE wonky file
-- The wonky file is hand updated.

SELECT 'EVENT', 'BOUT', 'OUTCOME', 'WEIGHTCLASS', 'METHOD', 'ROUND', 'TIME', 'TIME FORMAT', 'REFEREE', 'DETAILS', 'URL'
UNION ALL
select * from ufc_fight_results
WHERE NOT `TIME FORMAT` in ('3 Rnd (5-5-5)', '5 Rnd (5-5-5-5-5)', '2 Rnd (5-5)', '3 Rnd + OT (5-5-5-5)') AND Round <> '1'
INTO OUTFILE '/usr/var/ufc_stats/wonky_ufc_fight_times.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
*/



-- Step 1 Remove wonky rows from fight ufc_fight_results
DELETE FROM ufc_fight_results
WHERE NOT `TIME FORMAT` in ('3 Rnd (5-5-5)', '5 Rnd (5-5-5-5-5)', '2 Rnd (5-5)', '3 Rnd + OT (5-5-5-5)') AND Round <> '1';


-- Step 2 Import corrected wonky fight times
.import --csv --skip 1  'wonky_ufc_fight_times.csv' ufc_fight_results



/*
-- Manually assign weight index and sex classification to each weight class
*/

DROP INDEX IF EXISTS ufc_fight_results_WEIGHTCLASS;
CREATE INDEX ufc_fight_results_WEIGHTCLASS on ufc_fight_results(WEIGHTCLASS);

DROP TABLE if exists weightclass_lookup;
CREATE TABLE weightclass_lookup (
    weightclass VARCHAR(25)  PRIMARY KEY,
    weightindex INTEGER,
    weight INTEGER,
    sex INTEGER
);
INSERT INTO weightclass_lookup (weightclass, weightindex, weight, sex) VALUES
    ('Strawweight', '01', '115', '01'),
    ('Women''s Flyweight', '02', '125', '01'), 
    ('Women''s Bantamweight', '03', '135', '01'), 
    ('Women''s Featherweight', '04', '145', '01'), 
    ('Flyweight', '05', '125', '02'),
    ('Bantamweight', '06', '135', '02'), 
    ('Featherweight', '07', '145', '02'),
    ('Lightweight', '08', '155', '02'),
    ('Welterweight', '09', '170', '02'),
    ('Middleweight', '10', '185', '02'),
    ('Light Heavyweight', '11', '205', '02'),
    ('Heavyweight', '12', '265', '02');


DROP TABLE IF EXISTS ufc_weightclass_stats;

CREATE TABLE ufc_weightclass_stats as
SELECT DISTINCT 
    ufc_fight_results.WEIGHTCLASS,
    CAST(wl.weightindex AS UNSIGNED) AS weightindex,
    wl.weight AS weight,
    CAST(wl.sex AS UNSIGNED) AS sex
FROM ufc_fight_results
JOIN weightclass_lookup wl ON ufc_fight_results.WEIGHTCLASS LIKE '%' || wl.weightclass || '%'
ORDER BY weightindex;


-- Adding sex columns to other tables, along with indexes for performance

ALTER TABLE ufc_fight_results ADD COLUMN sex INT;
DROP INDEX IF EXISTS idx_ufc_fight_results_sex;
CREATE INDEX idx_ufc_fight_results_sex ON ufc_fight_results(sex);
DROP INDEX IF EXISTS idx_ufc_fight_results_weightclass;
CREATE INDEX idx_ufc_fight_results_weightclass ON ufc_fight_results(WEIGHTCLASS);
DROP INDEX IF EXISTS idx_ufc_weightclass_stats_weightclass;
CREATE INDEX idx_ufc_weightclass_stats_weightclass ON ufc_weightclass_stats(WEIGHTCLASS);

-- Add and index columns in ufc_fighter_tott table
ALTER TABLE ufc_fighter_tott ADD COLUMN sex INT;
DROP INDEX IF EXISTS idx_ufc_fighter_tott_sex;
CREATE INDEX idx_ufc_fighter_tott_sex ON ufc_fighter_tott(sex);
UPDATE ufc_fight_results
SET sex = (SELECT sex FROM ufc_weightclass_stats WHERE ufc_fight_results.WEIGHTCLASS = ufc_weightclass_stats.WEIGHTCLASS);
UPDATE ufc_fighter_tott
SET sex = (SELECT sex FROM ufc_fight_results 
           WHERE TRIM(ufc_fight_results.BOUT) LIKE '%' || TRIM(ufc_fighter_tott.FIGHTER) || '%');

-- Cleanup and remove any tott where the sex is unkown
DELETE FROM ufc_fighter_tott WHERE sex IS NULL;
ALTER TABLE ufc_fighter_tott ADD COLUMN weightindex INT;
ALTER TABLE ufc_fighter_tott ADD COLUMN weight_stat INT;

-- Merge Weightindex stats into Tott
DROP TABLE IF exists ufc_fighter_tott_new;
CREATE TABLE ufc_fighter_tott_new as SELECT * from ufc_fighter_tott WHERE 0;

With RankedRows AS (
SELECT  
t."FIGHTER" as FIGHTER,
t."HEIGHT",
t."WEIGHT",
t."REACH",
t."STANCE",
t."DOB",
t."URL",
t."sex",
uws.weightindex as weightindex,
uws.weight as weight_stat,
ROW_NUMBER() OVER (PARTITION BY t."FIGHTER" ORDER BY ABS(uws.weight - t."WEIGHT") , (uws.weight - t."WEIGHT") DESC) AS rn
FROM ufc_fighter_tott t JOIN ufc_weightclass_stats uws
ON t.sex = uws.sex
)
INSERT into ufc_fighter_tott_new ("FIGHTER","HEIGHT","WEIGHT","REACH","STANCE","DOB","URL","sex","weight_stat","weightindex")
SELECT "FIGHTER","HEIGHT","WEIGHT","REACH","STANCE","DOB","URL","sex","weight_stat","weightindex" FROM "RankedRows" where rn=1 LIMIT 4000;
-- Weird issue default limit to 100
select count(*) from ufc_fighter_tott_new;
DROP TABLE IF EXISTS ufc_fighter_tott;
ALTER TABLE ufc_fighter_tott_new
  RENAME TO ufc_fighter_tott;


ALTER TABLE ufc_fight_results
ADD COLUMN weightindex INT;

UPDATE ufc_fight_results
SET 
weightindex =(SELECT w.weightindex FROM ufc_weightclass_stats w WHERE w."WEIGHTCLASS" = ufc_fight_results."WEIGHTCLASS");

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
SUM(REV) as rev,
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

/* Big aggregation query to look at past performance of fighters (e.g. past 3 fights, past 5 fights) */
DROP VIEW IF EXISTS single_full_view;
-- Make sure to change the LOCATE FLAG in the WHERE Clause to the correct fighter
CREATE VIEW single_full_view AS
SELECT
    date(ed.DATE) AS DATE,
    ms.jbout,
    ms.jevent,
    ms.jfighter,
    ms.EVENT,
    ms.BOUT,
    ms.FIGHTER,
    w.fight_time_minutes,
    IFNULL(SUM(fight_time_minutes) OVER wu1, 0) AS precomp_tot_time_in_cage,
    IFNULL(SUM(fight_time_minutes) OVER wu0, 0) AS postcomp_tot_time_in_cage,
    (julianday(date(ed.DATE)) - julianday(date(t.DOB))) / 365 AS age,
    t.HEIGHT,
    t.WEIGHT,
    t.REACH,
    date(t.DOB) AS DOB,
    t.sex,
    t.weightindex,
    t.weight_stat,
    r.weightindex AS weight_of_fight,
    IFNULL(AVG(r.weightindex) OVER w31, 0) AS weight_avg3,
    w.win,
    w.loss,
    w.udec,
    w.udecd,
    w.mdec,
    w.mdecd,
    w.sdec,
    w.sdecd,
    w.subw,
    w.subwd,
    w.ko,
    w.kod,
    ms.subatt,
    IFNULL(SUM(fight_time_minutes) OVER w20, 0) AS postcomp_tot_time_in_cage_3,
    IFNULL(SUM(fight_time_minutes) OVER w31, 0) AS precomp_tot_time_in_cage_3,
    IFNULL(SUM(fight_time_minutes) OVER w40, 0) AS postcomp_tot_time_in_cage_5,
    IFNULL(SUM(fight_time_minutes) OVER w51, 0) AS precomp_tot_time_in_cage_5,
    -- SLpM: Significant Strikes Landed per Minute over 3 and 5 fights
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER wu0, 0) / IFNULL(SUM(fight_time_minutes) OVER wu0, 0)
    END AS postcomp_sigstr_pm,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER wu1, 0) / IFNULL(SUM(fight_time_minutes) OVER wu1, 0)
    END AS precomp_sigstr_pm,
    -- 3 fights
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER w20, 0) / IFNULL(SUM(fight_time_minutes) OVER w20, 0)
    END AS postcomp_sigstr_pm3,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER w31, 0) / IFNULL(SUM(fight_time_minutes) OVER w31, 0)
    END AS precomp_sigstr_pm3,
    -- 5 fights
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER w40, 0) / IFNULL(SUM(fight_time_minutes) OVER w40, 0)
    END AS postcomp_sigstr_pm5,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER w51, 0) / IFNULL(SUM(fight_time_minutes) OVER w51, 0)
    END AS precomp_sigstr_pm5,
    -- TDAvg: average take downs landed per 15 minutes over 3 and 5 fights
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER wu0, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER wu0, 0)
    END AS postcomp_tdavg,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER wu1, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER wu1, 0)
    END AS precomp_tdavg,
    -- 3 fight average
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER w20, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER w20, 0)
    END AS postcomp_tdavg3,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER w31, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER w31, 0)
    END AS precomp_tdavg3,
    -- 5 fights
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER w40, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER w40, 0)
    END AS postcomp_tdavg5,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER w51, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER w51, 0)
    END AS precomp_tdavg5,
    -- SApM: Significant Strikes Absorbed per Minute over 3 and 5 fights
    sa.sigstrabs,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstrabs) OVER wu0, 0) / IFNULL(SUM(fight_time_minutes) OVER wu0, 0)
    END AS postcomp_sapm,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstrabs) OVER wu1, 0) / IFNULL(SUM(fight_time_minutes) OVER wu1, 0)
    END AS precomp_sapm,
    -- 3 fights
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstrabs) OVER w31, 0) / IFNULL(SUM(fight_time_minutes) OVER w31, 0)
    END AS precomp_sapm3,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstrabs) OVER w20, 0) / IFNULL(SUM(fight_time_minutes) OVER w20, 0)
    END AS postcomp_sapm3,
    -- 5 fights
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstrabs) OVER w51, 0) / IFNULL(SUM(fight_time_minutes) OVER w51, 0)
    END AS precomp_sapm5,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstrabs) OVER w40, 0) / IFNULL(SUM(fight_time_minutes) OVER w40, 0)
    END AS postcomp_sapm5,
    -- SubAvg: average submission attempts per 15 minutes over 3 and 5 fights
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(subatt) OVER wu0, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER wu0, 0)
    END AS postcomp_subavg,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(subatt) OVER wu1, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER wu1, 0)
    END AS precomp_subavg,
    -- 3 fight average
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(subatt) OVER w20, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER w20, 0)
    END AS postcomp_subavg3,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(subatt) OVER w31, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER w31, 0)
    END AS precomp_subavg3,
    -- 5 fights
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(subatt) OVER w40, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER w40, 0)
    END AS postcomp_subavg5,
    CASE 
        WHEN IFNULL(SUM(fight_time_minutes) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(subatt) OVER w51, 0) * 15 / IFNULL(SUM(fight_time_minutes) OVER w51, 0)
    END AS precomp_subavg5,
    -- TdDef: Takedown Defense
    sa.tdabs,
    sa.tdattfromopp,
    CASE 
        WHEN IFNULL(SUM(tdattfromopp) OVER wu0, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(tdabs) OVER wu0, 0) / IFNULL(SUM(tdattfromopp) OVER wu0, 0))
    END AS postcomp_tddef,
    CASE 
        WHEN IFNULL(SUM(tdattfromopp) OVER wu1, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(tdabs) OVER wu1, 0) / IFNULL(SUM(tdattfromopp) OVER wu1, 0))
    END AS precomp_tddef,
    -- 3 fights
    CASE 
        WHEN IFNULL(SUM(tdattfromopp) OVER w20, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(tdabs) OVER w20, 0) / IFNULL(SUM(tdattfromopp) OVER w20, 0))
    END AS postcomp_tddef3,
    CASE 
        WHEN IFNULL(SUM(tdattfromopp) OVER w31, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(tdabs) OVER w31, 0) / IFNULL(SUM(tdattfromopp) OVER w31, 0))
    END AS precomp_tddef3,
    -- 5 fights
    CASE 
        WHEN IFNULL(SUM(tdattfromopp) OVER w40, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(tdabs) OVER w40, 0) / IFNULL(SUM(tdattfromopp) OVER w40, 0))
    END AS postcomp_tddef5,
    CASE 
        WHEN IFNULL(SUM(tdattfromopp) OVER w51, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(tdabs) OVER w51, 0) / IFNULL(SUM(tdattfromopp) OVER w51, 0))
    END AS precomp_tddef5,
    -- Str. Acc. Significant Striking Accuracy with average of 3 and 5 fights
    ms.sigstracc AS ostrabs,
    IFNULL(SUM(sigstracc) OVER wu0, 0) AS postcomp_ostrabssum,
    ms.sigstracc,
    ms.sigstratt,
    IFNULL(SUM(sigstracc) OVER wu1, 0) AS precomp_sigstraccsum,
    IFNULL(SUM(sigstracc) OVER wu0, 0) AS postcomp_sigstraccsum,
    CASE 
        WHEN IFNULL(SUM(sigstratt) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER wu0, 0) / IFNULL(SUM(sigstratt) OVER wu0, 0)
    END AS postcomp_sigstr_perc,
    CASE 
        WHEN IFNULL(SUM(sigstratt) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER wu1, 0) / IFNULL(SUM(sigstratt) OVER wu1, 0)
    END AS precomp_sigstr_perc,
    -- 3 fight average
    CASE 
        WHEN IFNULL(SUM(sigstratt) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER w31, 0) / IFNULL(SUM(sigstratt) OVER w31, 0)
    END AS precomp_sigstr_perc3,
    CASE 
        WHEN IFNULL(SUM(sigstratt) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER w20, 0) / IFNULL(SUM(sigstratt) OVER w20, 0)
    END AS postcomp_sigstr_perc3,
    -- 5 fight average
    CASE 
        WHEN IFNULL(SUM(sigstratt) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER w51, 0) / IFNULL(SUM(sigstratt) OVER w51, 0)
    END AS precomp_sigstr_perc5,
    CASE 
        WHEN IFNULL(SUM(sigstratt) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(sigstracc) OVER w40, 0) / IFNULL(SUM(sigstratt) OVER w40, 0)
    END AS postcomp_sigstr_perc5,
    -- StrDef: Striking Defense 
    sa.sigstrattfromopp,
    CASE 
        WHEN IFNULL(SUM(sa.sigstrattfromopp) OVER wu0, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(sigstrabs) OVER wu0, 0) / IFNULL(SUM(sigstrattfromopp) OVER wu0, 0))
    END AS postcomp_strdef,
    CASE 
        WHEN IFNULL(SUM(sa.sigstrattfromopp) OVER wu1, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(sigstrabs) OVER wu1, 0) / IFNULL(SUM(sigstrattfromopp) OVER wu1, 0))
    END AS precomp_strdef,
    -- 3 fights
    CASE 
        WHEN IFNULL(SUM(sa.sigstrattfromopp) OVER w20, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(sigstrabs) OVER w20, 0) / IFNULL(SUM(sigstrattfromopp) OVER w20, 0))
    END AS postcomp_strdef3,
    CASE 
        WHEN IFNULL(SUM(sa.sigstrattfromopp) OVER w31, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(sigstrabs) OVER w31, 0) / IFNULL(SUM(sigstrattfromopp) OVER w31, 0))
    END AS precomp_strdef3,
    -- 5 fights
    CASE 
        WHEN IFNULL(SUM(sa.sigstrattfromopp) OVER w40, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(sigstrabs) OVER w40, 0) / IFNULL(SUM(sigstrattfromopp) OVER w40, 0))
    END AS postcomp_strdef5,
    CASE 
        WHEN IFNULL(SUM(sa.sigstrattfromopp) OVER w51, 0) = 0 THEN 0
        ELSE 1 - (IFNULL(SUM(sigstrabs) OVER w51, 0) / IFNULL(SUM(sigstrattfromopp) OVER w51, 0))
    END AS precomp_strdef5,
    -- TD Acc. Take Down Accuracy with average of 3 and 5 fights
    ms.tdacc,
    ms.tdatt,
    CASE 
        WHEN IFNULL(SUM(tdatt) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER wu0, 0) / IFNULL(SUM(tdatt) OVER wu0, 0)
    END AS postcomp_tdacc_perc,
    CASE 
        WHEN IFNULL(SUM(tdatt) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER wu1, 0) / IFNULL(SUM(tdatt) OVER wu1, 0)
    END AS precomp_tdacc_perc,
    -- 3 fight average
    CASE 
        WHEN IFNULL(SUM(tdatt) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER w31, 0) / IFNULL(SUM(tdatt) OVER w31, 0)
    END AS precomp_tdacc_perc3,
    CASE 
        WHEN IFNULL(SUM(tdatt) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER w20, 0) / IFNULL(SUM(tdatt) OVER w20, 0)
    END AS postcomp_tdacc_perc3,
    -- 5 fight average
    CASE 
        WHEN IFNULL(SUM(tdatt) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER w51, 0) / IFNULL(SUM(tdatt) OVER w51, 0)
    END AS precomp_tdacc_perc5,
    CASE 
        WHEN IFNULL(SUM(tdatt) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(tdacc) OVER w40, 0) / IFNULL(SUM(tdatt) OVER w40, 0)
    END AS postcomp_tdacc_perc5,
    -- Total strike accuracy over 3 and 5 fights
    ms.totalacc,
    ms.totalatt,
    CASE 
        WHEN IFNULL(SUM(totalatt) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(totalacc) OVER wu0, 0) / IFNULL(SUM(totalatt) OVER wu0, 0)
    END AS postcomp_totalacc_perc,
    CASE 
        WHEN IFNULL(SUM(totalatt) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(totalacc) OVER wu1, 0) / IFNULL(SUM(totalatt) OVER wu1, 0)
    END AS precomp_totalacc_perc,
    -- 3 fight average
    CASE 
        WHEN IFNULL(SUM(totalatt) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(totalacc) OVER w31, 0) / IFNULL(SUM(totalatt) OVER w31, 0)
    END AS precomp_totalacc_perc3,
    CASE 
        WHEN IFNULL(SUM(totalatt) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(totalacc) OVER w20, 0) / IFNULL(SUM(totalatt) OVER w20, 0)
    END AS postcomp_totalacc_perc3,
    -- 5 fight average
    CASE 
        WHEN IFNULL(SUM(totalatt) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(totalacc) OVER w51, 0) / IFNULL(SUM(totalatt) OVER w51, 0)
    END AS precomp_totalacc_perc5,
    CASE 
        WHEN IFNULL(SUM(totalatt) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(totalacc) OVER w40, 0) / IFNULL(SUM(totalatt) OVER w40, 0)
    END AS postcomp_totalacc_perc5,
    -- average head strike accuracy over 3 and 5 fights
    ms.headacc,
    ms.headatt,
    CASE 
        WHEN IFNULL(SUM(headatt) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(headacc) OVER wu0, 0) / IFNULL(SUM(headatt) OVER wu0, 0)
    END AS postcomp_headacc_perc,
    CASE 
        WHEN IFNULL(SUM(headatt) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(headacc) OVER wu1, 0) / IFNULL(SUM(headatt) OVER wu1, 0)
    END AS precomp_headacc_perc,
    -- 3 fight average
    CASE 
        WHEN IFNULL(SUM(headatt) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(headacc) OVER w31, 0) / IFNULL(SUM(headatt) OVER w31, 0)
    END AS precomp_headacc_perc3,
    CASE 
        WHEN IFNULL(SUM(headatt) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(headacc) OVER w20, 0) / IFNULL(SUM(headatt) OVER w20, 0)
    END AS postcomp_headacc_perc3,
    -- 5 fight average
    CASE 
        WHEN IFNULL(SUM(headatt) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(headacc) OVER w51, 0) / IFNULL(SUM(headatt) OVER w51, 0)
    END AS precomp_headacc_perc5,
    CASE 
        WHEN IFNULL(SUM(headatt) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(headacc) OVER w40, 0) / IFNULL(SUM(headatt) OVER w40, 0)
    END AS postcomp_headacc_perc5,
    -- average body strike accuracy over 3 and 5 fights
    ms.bodyacc,
    ms.bodyatt,
    CASE 
        WHEN IFNULL(SUM(bodyatt) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(bodyacc) OVER wu0, 0) / IFNULL(SUM(bodyatt) OVER wu0, 0)
    END AS postcomp_bodyacc_perc,
    CASE 
        WHEN IFNULL(SUM(bodyatt) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(bodyacc) OVER wu1, 0) / IFNULL(SUM(bodyatt) OVER wu1, 0)
    END AS precomp_bodyacc_perc,
    -- 3 fight average
    CASE 
        WHEN IFNULL(SUM(bodyatt) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(bodyacc) OVER w31, 0) / IFNULL(SUM(bodyatt) OVER w31, 0)
    END AS precomp_bodyacc_perc3,
    CASE 
        WHEN IFNULL(SUM(bodyatt) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(bodyacc) OVER w20, 0) / IFNULL(SUM(bodyatt) OVER w20, 0)
    END AS postcomp_bodyacc_perc3,
    -- 5 fight average
    CASE 
        WHEN IFNULL(SUM(bodyatt) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(bodyacc) OVER w51, 0) / IFNULL(SUM(bodyatt) OVER w51, 0)
    END AS precomp_bodyacc_perc5,
    CASE 
        WHEN IFNULL(SUM(bodyatt) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(bodyacc) OVER w40, 0) / IFNULL(SUM(bodyatt) OVER w40, 0)
    END AS postcomp_bodyacc_perc5,
    -- average leg strike accuracy over 3 and 5 fights
    ms.legacc,
    ms.legatt,
    CASE 
        WHEN IFNULL(SUM(legatt) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(legacc) OVER wu0, 0) / IFNULL(SUM(legatt) OVER wu0, 0)
    END AS postcomp_legacc_perc,
    CASE 
        WHEN IFNULL(SUM(legatt) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(legacc) OVER wu1, 0) / IFNULL(SUM(legatt) OVER wu1, 0)
    END AS precomp_legacc_perc,
    -- 3 fight average
    CASE 
        WHEN IFNULL(SUM(legatt) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(legacc) OVER w31, 0) / IFNULL(SUM(legatt) OVER w31, 0)
    END AS precomp_legacc_perc3,
    CASE 
        WHEN IFNULL(SUM(legatt) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(legacc) OVER w20, 0) / IFNULL(SUM(legatt) OVER w20, 0)
    END AS postcomp_legacc_perc3,
    -- 5 fight average
    CASE 
        WHEN IFNULL(SUM(legatt) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(legacc) OVER w51, 0) / IFNULL(SUM(legatt) OVER w51, 0)
    END AS precomp_legacc_perc5,
    CASE 
        WHEN IFNULL(SUM(legatt) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(legacc) OVER w40, 0) / IFNULL(SUM(legatt) OVER w40, 0)
    END AS postcomp_legacc_perc5,
    -- average distance strike accuracy over 3 and 5 fights
    ms.distacc,
    ms.distatt,
    CASE 
        WHEN IFNULL(SUM(distatt) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(distacc) OVER wu0, 0) / IFNULL(SUM(distatt) OVER wu0, 0)
    END AS postcomp_distacc_perc,
    CASE 
        WHEN IFNULL(SUM(distatt) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(distacc) OVER wu1, 0) / IFNULL(SUM(distatt) OVER wu1, 0)
    END AS precomp_distacc_perc,
    -- 3 fight average
    CASE 
        WHEN IFNULL(SUM(distatt) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(distacc) OVER w31, 0) / IFNULL(SUM(distatt) OVER w31, 0)
    END AS precomp_distacc_perc3,
    CASE 
        WHEN IFNULL(SUM(distatt) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(distacc) OVER w20, 0) / IFNULL(SUM(distatt) OVER w20, 0)
    END AS postcomp_distacc_perc3,
    -- 5 fight average
    CASE 
        WHEN IFNULL(SUM(distatt) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(distacc) OVER w51, 0) / IFNULL(SUM(distatt) OVER w51, 0)
    END AS precomp_distacc_perc5,
    CASE 
        WHEN IFNULL(SUM(distatt) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(distacc) OVER w40, 0) / IFNULL(SUM(distatt) OVER w40, 0)
    END AS postcomp_distacc_perc5,
    -- average clinch strike accuracy over 3 and 5 fights
    ms.clinchacc,
    ms.clinchatt,
    CASE 
        WHEN IFNULL(SUM(clinchatt) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(clinchacc) OVER wu0, 0) / IFNULL(SUM(clinchatt) OVER wu0, 0)
    END AS postcomp_clinchacc_perc,
    CASE 
        WHEN IFNULL(SUM(clinchatt) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(clinchacc) OVER wu1, 0) / IFNULL(SUM(clinchatt) OVER wu1, 0)
    END AS precomp_clinchacc_perc,
    -- 3 fight average
    CASE 
        WHEN IFNULL(SUM(clinchatt) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(clinchacc) OVER w31, 0) / IFNULL(SUM(clinchatt) OVER w31, 0)
    END AS precomp_clinchacc_perc3,
    CASE 
        WHEN IFNULL(SUM(clinchatt) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(clinchacc) OVER w20, 0) / IFNULL(SUM(clinchatt) OVER w20, 0)
    END AS postcomp_clinchacc_perc3,
    -- 5 fight average
    CASE 
        WHEN IFNULL(SUM(clinchatt) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(clinchacc) OVER w51, 0) / IFNULL(SUM(clinchatt) OVER w51, 0)
    END AS precomp_clinchacc_perc5,
    CASE 
        WHEN IFNULL(SUM(clinchatt) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(clinchacc) OVER w40, 0) / IFNULL(SUM(clinchatt) OVER w40, 0)
    END AS postcomp_clinchacc_perc5,
    -- average ground strike accuracy over 3 and 5 fights
    ms.groundacc,
    ms.groundatt,
    CASE 
        WHEN IFNULL(SUM(groundatt) OVER wu0, 0) = 0 THEN 0
        ELSE IFNULL(SUM(groundacc) OVER wu0, 0) / IFNULL(SUM(groundatt) OVER wu0, 0)
    END AS postcomp_groundacc_perc,
    CASE 
        WHEN IFNULL(SUM(groundatt) OVER wu1, 0) = 0 THEN 0
        ELSE IFNULL(SUM(groundacc) OVER wu1, 0) / IFNULL(SUM(groundatt) OVER wu1, 0)
    END AS precomp_groundacc_perc,
    -- 3 fight average
    CASE 
        WHEN IFNULL(SUM(groundatt) OVER w20, 0) = 0 THEN 0
        ELSE IFNULL(SUM(groundacc) OVER w20, 0) / IFNULL(SUM(groundatt) OVER w20, 0)
    END AS postcomp_groundacc_perc3,
    CASE 
        WHEN IFNULL(SUM(groundatt) OVER w31, 0) = 0 THEN 0
        ELSE IFNULL(SUM(groundacc) OVER w31, 0) / IFNULL(SUM(groundatt) OVER w31, 0)
    END AS precomp_groundacc_perc3,
    -- 5 fight average
    CASE 
        WHEN IFNULL(SUM(groundatt) OVER w40, 0) = 0 THEN 0
        ELSE IFNULL(SUM(groundacc) OVER w40, 0) / IFNULL(SUM(groundatt) OVER w40, 0)
    END AS postcomp_groundacc_perc5,
    CASE 
        WHEN IFNULL(SUM(groundatt) OVER w51, 0) = 0 THEN 0
        ELSE IFNULL(SUM(groundacc) OVER w51, 0) / IFNULL(SUM(groundatt) OVER w51, 0)
    END AS precomp_groundacc_perc5,
    -- win/loss/ko/kdo over 3 and 5 fights
    IFNULL(SUM(win) OVER wu0, 0) AS postcomp_winsum,
    IFNULL(SUM(win) OVER wu1, 0) AS precomp_winsum,
    IFNULL(COUNT(*) OVER wu0, 0) AS postcomp_boutcount,
    IFNULL(COUNT(*) OVER wu1, 0) AS precomp_boutcount,
    IFNULL(AVG(win) OVER wu0, 0) AS postcomp_winavg,
    IFNULL(AVG(win) OVER wu1, 0) AS precomp_winavg,
    IFNULL(SUM(win) OVER w20, 0) AS postcomp_winsum3,
    IFNULL(SUM(win) OVER w31, 0) AS precomp_winsum3,
    IFNULL(AVG(win) OVER w20, 0) AS postcomp_winavg3,
    IFNULL(AVG(win) OVER w31, 0) AS precomp_winavg3,
    IFNULL(SUM(win) OVER w40, 0) AS postcomp_winsum5,
    IFNULL(SUM(win) OVER w51, 0) AS precomp_winsum5,
    IFNULL(AVG(win) OVER w40, 0) AS postcomp_winavg5,
    IFNULL(AVG(win) OVER w51, 0) AS precomp_winavg5,
    IFNULL(SUM(loss) OVER wu0, 0) AS postcomp_losssum,
    IFNULL(SUM(loss) OVER wu1, 0) AS precomp_losssum,
    IFNULL(AVG(loss) OVER wu0, 0) AS postcomp_lossavg,
    IFNULL(AVG(loss) OVER wu1, 0) AS precomp_lossavg,
    IFNULL(SUM(loss) OVER w20, 0) AS postcomp_losssum3,
    IFNULL(SUM(loss) OVER w31, 0) AS precomp_losssum3,
    IFNULL(AVG(loss) OVER w20, 0) AS postcomp_lossavg3,
    IFNULL(AVG(loss) OVER w31, 0) AS precomp_lossavg3,
    IFNULL(SUM(loss) OVER w40, 0) AS postcomp_losssum5,
    IFNULL(SUM(loss) OVER w51, 0) AS precomp_losssum5,
    IFNULL(AVG(loss) OVER w40, 0) AS postcomp_lossavg5,
    IFNULL(AVG(loss) OVER w51, 0) AS precomp_lossavg5,
    IFNULL(SUM(ko) OVER wu0, 0) AS postcomp_kosum,
    IFNULL(SUM(ko) OVER wu1, 0) AS precomp_kosum,
    IFNULL(AVG(ko) OVER wu0, 0) AS postcomp_koavg,
    IFNULL(AVG(ko) OVER wu1, 0) AS precomp_koavg,
    IFNULL(SUM(ko) OVER w20, 0) AS postcomp_kosum3,
    IFNULL(SUM(ko) OVER w31, 0) AS precomp_kosum3,
    IFNULL(AVG(ko) OVER w20, 0) AS postcomp_koavg3,
    IFNULL(AVG(ko) OVER w31, 0) AS precomp_koavg3,
    IFNULL(SUM(ko) OVER w40, 0) AS postcomp_kosum5,
    IFNULL(SUM(ko) OVER w51, 0) AS precomp_kosum5,
    IFNULL(AVG(ko) OVER w40, 0) AS postcomp_koavg5,
    IFNULL(AVG(ko) OVER w51, 0) AS precomp_koavg5,
    IFNULL(SUM(kod) OVER wu0, 0) AS postcomp_kodsum,
    IFNULL(SUM(kod) OVER wu1, 0) AS precomp_kodsum,
    IFNULL(AVG(kod) OVER wu0, 0) AS postcomp_kodavg,
    IFNULL(AVG(kod) OVER wu1, 0) AS precomp_kodavg,
    IFNULL(SUM(kod) OVER w20, 0) AS postcomp_kodsum3,
    IFNULL(SUM(kod) OVER w31, 0) AS precomp_kodsum3,
    IFNULL(AVG(kod) OVER w20, 0) AS postcomp_kodavg3,
    IFNULL(AVG(kod) OVER w31, 0) AS precomp_kodavg3,
    IFNULL(SUM(kod) OVER w40, 0) AS postcomp_kodsum5,
    IFNULL(SUM(kod) OVER w51, 0) AS precomp_kodsum5,
    IFNULL(AVG(kod) OVER w40, 0) AS postcomp_kodavg5,
    IFNULL(AVG(kod) OVER w51, 0) AS precomp_kodavg5,
    IFNULL(SUM(subw) OVER wu0, 0) AS postcomp_subwsum,
    IFNULL(SUM(subw) OVER wu1, 0) AS precomp_subwsum,
    IFNULL(AVG(subw) OVER wu0, 0) AS postcomp_subwavg,
    IFNULL(AVG(subw) OVER wu1, 0) AS precomp_subwavg,
    IFNULL(SUM(subw) OVER w20, 0) AS postcomp_subwsum3,
    IFNULL(SUM(subw) OVER w31, 0) AS precomp_subwsum3,
    IFNULL(AVG(subw) OVER w20, 0) AS postcomp_subwavg3,
    IFNULL(AVG(subw) OVER w31, 0) AS precomp_subwavg3,
    IFNULL(SUM(subw) OVER w40, 0) AS postcomp_subwsum5,
    IFNULL(SUM(subw) OVER w51, 0) AS precomp_subwsum5,
    IFNULL(AVG(subw) OVER w40, 0) AS postcomp_subwavg5,
    IFNULL(AVG(subw) OVER w51, 0) AS precomp_subwavg5,
    IFNULL(SUM(subwd) OVER wu0, 0) AS postcomp_subwdsum,
    IFNULL(SUM(subwd) OVER wu1, 0) AS precomp_subwdsum,
    IFNULL(AVG(subwd) OVER wu0, 0) AS postcomp_subwdavg,
    IFNULL(AVG(subwd) OVER wu1, 0) AS precomp_subwdavg,
    IFNULL(SUM(subwd) OVER w20, 0) AS postcomp_subwdsum3,
    IFNULL(SUM(subwd) OVER w31, 0) AS precomp_subwdsum3,
    IFNULL(AVG(subwd) OVER w20, 0) AS postcomp_subwdavg3,
    IFNULL(AVG(subwd) OVER w31, 0) AS precomp_subwdavg3,
    IFNULL(SUM(subwd) OVER w40, 0) AS postcomp_subwdsum5,
    IFNULL(SUM(subwd) OVER w51, 0) AS precomp_subwdsum5,
    IFNULL(AVG(subwd) OVER w40, 0) AS postcomp_subwdavg5,
    IFNULL(AVG(subwd) OVER w51, 0) AS precomp_subwdavg5,
    IFNULL(SUM(udec) OVER wu0, 0) AS postcomp_udecsum,
    IFNULL(SUM(udec) OVER wu1, 0) AS precomp_udecsum,
    IFNULL(AVG(udec) OVER wu0, 0) AS postcomp_udecavg,
    IFNULL(AVG(udec) OVER wu1, 0) AS precomp_udecavg,
    IFNULL(SUM(udec) OVER w20, 0) AS postcomp_udecsum3,
    IFNULL(SUM(udec) OVER w31, 0) AS precomp_udecsum3,
    IFNULL(AVG(udec) OVER w20, 0) AS postcomp_udecavg3,
    IFNULL(AVG(udec) OVER w31, 0) AS precomp_udecavg3,
    IFNULL(SUM(udec) OVER w40, 0) AS postcomp_udecsum5,
    IFNULL(SUM(udec) OVER w51, 0) AS precomp_udecsum5,
    IFNULL(AVG(udec) OVER w40, 0) AS postcomp_udecavg5,
    IFNULL(AVG(udec) OVER w51, 0) AS precomp_udecavg5,
    IFNULL(SUM(udecd) OVER wu0, 0) AS postcomp_udecdsum,
    IFNULL(SUM(udecd) OVER wu1, 0) AS precomp_udecdsum,
    IFNULL(AVG(udecd) OVER wu0, 0) AS postcomp_udecdavg,
    IFNULL(AVG(udecd) OVER wu1, 0) AS precomp_udecdavg,
    IFNULL(SUM(udecd) OVER w20, 0) AS postcomp_udecdsum3,
    IFNULL(SUM(udecd) OVER w31, 0) AS precomp_udecdsum3,
    IFNULL(AVG(udecd) OVER w20, 0) AS postcomp_udecdavg3,
    IFNULL(AVG(udecd) OVER w31, 0) AS precomp_udecdavg3,
    IFNULL(SUM(udecd) OVER w40, 0) AS postcomp_udecdsum5,
    IFNULL(SUM(udecd) OVER w51, 0) AS precomp_udecdsum5,
    IFNULL(AVG(udecd) OVER w40, 0) AS postcomp_udecdavg5,
    IFNULL(AVG(udecd) OVER w51, 0) AS precomp_udecdavg5
FROM
    ufc_fighter_match_stats ms,
    ufc_event_details ed,
    ufc_winlossko w,
    ufc_fighter_tott t,
    ufc_fight_results r,
    (
        SELECT
            DISTINCT
            TRIM(REPLACE(fighter_stats.EVENT, ' ', '')) AS jfs_event,
            TRIM(REPLACE(fighter_stats.FIGHTER, ' ', '')) AS jfs_fighter,
            TRIM(REPLACE(fighter_stats.BOUT, ' ', '')) AS jfs_bout,
            opponent_stats.sigstracc AS sigstrabs,
            opponent_stats.sigstratt AS sigstrattfromopp,
            opponent_stats.tdacc AS tdabs,
            opponent_stats.tdatt AS tdattfromopp
        FROM
            ufc_fighter_match_stats fighter_stats
        JOIN
            ufc_fighter_match_stats opponent_stats
        ON
            fighter_stats.jbout = opponent_stats.jbout
            AND fighter_stats.jfighter != opponent_stats.jfighter
            AND fighter_stats.jevent = opponent_stats.jevent
    ) sa
WHERE
    ms.jevent = ed.jevent
    AND ms.jevent = w.jevent
    AND ms.jfighter = w.jfighter
    AND ms.jfighter = t.jfighter
    AND ms.jevent = r.jevent
    AND ms.jbout = r.jbout
    AND ms.jevent = sa.jfs_event
    AND ms.jfighter = sa.jfs_fighter
    AND ms.jbout = sa.jfs_bout
    AND instr(ms.jbout,ms.jfighter) > 0
WINDOW
    w20 AS (PARTITION BY ms.jfighter ORDER BY date(ed.DATE) ROWS BETWEEN 2 PRECEDING AND 0 PRECEDING),
    w31 AS (PARTITION BY ms.jfighter ORDER BY date(ed.DATE) ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING),
    w40 AS (PARTITION BY ms.jfighter ORDER BY date(ed.DATE) ROWS BETWEEN 4 PRECEDING AND 0 PRECEDING),
    w51 AS (PARTITION BY ms.jfighter ORDER BY date(ed.DATE) ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
    wu AS (PARTITION BY ms.jfighter ORDER BY date(ed.DATE) ROWS UNBOUNDED PRECEDING),
    wu0 AS (PARTITION BY ms.jfighter ORDER BY date(ed.DATE) ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING),
    wu1 AS (PARTITION BY ms.jfighter ORDER BY date(ed.DATE) ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);


/* JOIN and pull over Weigth Reach,reach from ufc_fighter_tott. */
DROP VIEW IF EXISTS single_fighter_view;
DROP VIEW IF EXISTS single_opponent_view;
CREATE VIEW single_fighter_view AS
SELECT * from single_full_view
WHERE INSTR(jbout,jfighter) = 1
ORDER BY DATE DESC;
CREATE VIEW single_opponent_view AS
SELECT * from single_full_view
WHERE INSTR(jbout,jfighter) > 1
ORDER BY DATE DESC;

DROP VIEW IF EXISTS pre_final;
Create View pre_final as
SELECT
'DATE',
'jbout',
'jevent',
'jfighter',
'EVENT',
'BOUT',
'FIGHTER',
'fight_time_minutes',
'precomp_tot_time_in_cage',
'postcomp_tot_time_in_cage',
'age',
'HEIGHT',
'WEIGHT',
'REACH',
'DOB',
'sex',
'weightindex',
'weight_stat',
'weight_of_fight',
'weight_avg3',
'win',
'loss',
'ko',
'kod',
'sub',
'subd',
'udec',
'udecd',
'subatt',
'postcomp_tot_time_in_cage_3',
'precomp_tot_time_in_cage_3',
'postcomp_tot_time_in_cage_5',
'precomp_tot_time_in_cage_5',
'postcomp_sigstr_pm',
'precomp_sigstr_pm',
'postcomp_sigstr_pm3',
'precomp_sigstr_pm3',
'postcomp_sigstr_pm5',
'precomp_sigstr_pm5',
'postcomp_tdavg',
'precomp_tdavg',
'postcomp_tdavg3',
'precomp_tdavg3',
'postcomp_tdavg5',
'precomp_tdavg5',
'sigstrabs',
'postcomp_sapm',
'precomp_sapm',
'precomp_sapm3',
'postcomp_sapm3',
'precomp_sapm5',
'postcomp_sapm5',
'postcomp_subavg',
'precomp_subavg',
'postcomp_subavg3',
'precomp_subavg3',
'postcomp_subavg5',
'precomp_subavg5',
'tdabs',
'tdattfromopp',
'postcomp_tddef',
'precomp_tddef',
'postcomp_tddef3',
'precomp_tddef3',
'postcomp_tddef5',
'precomp_tddef5',
'ostrabs',
'postcomp_ostrabssum',
'sigstracc',
'sigstratt',
'precomp_sigstraccsum',
'postcomp_sigstraccsum',
'postcomp_sigstr_perc',
'precomp_sigstr_perc',
'precomp_sigstr_perc3',
'postcomp_sigstr_perc3',
'precomp_sigstr_perc5',
'postcomp_sigstr_perc5',
'sigstrattfromopp',
'postcomp_strdef',
'precomp_strdef',
'postcomp_strdef3',
'precomp_strdef3',
'postcomp_strdef5',
'precomp_strdef5',
'tdacc',
'tdatt',
'postcomp_tdacc_perc',
'precomp_tdacc_perc',
'postcomp_tdacc_perc3',
'precomp_tdacc_perc3',
'postcomp_tdacc_perc5',
'precomp_tdacc_perc5',
'totalacc',
'totalatt',
'postcomp_totalacc_perc',
'precomp_totalacc_perc',
'postcomp_totalacc_perc3',
'precomp_totalacc_perc3',
'postcomp_totalacc_perc5',
'precomp_totalacc_perc5',
'headacc',
'headatt',
'postcomp_headacc_perc',
'precomp_headacc_perc',
'postcomp_headacc_perc3',
'precomp_headacc_perc3',
'postcomp_headacc_perc5',
'precomp_headacc_perc5',
'bodyacc',
'bodyatt',
'postcomp_bodyacc_perc',
'precomp_bodyacc_perc',
'postcomp_bodyacc_perc3',
'precomp_bodyacc_perc3',
'postcomp_bodyacc_perc5',
'precomp_bodyacc_perc5',
'legacc',
'legatt',
'postcomp_legacc_perc',
'precomp_legacc_perc',
'postcomp_legacc_perc3',
'precomp_legacc_perc3',
'postcomp_legacc_perc5',
'precomp_legacc_perc5',
'distacc',
'distatt',
'postcomp_distacc_perc',
'precomp_distacc_perc',
'postcomp_distacc_perc3',
'precomp_distacc_perc3',
'postcomp_distacc_perc5',
'precomp_distacc_perc5',
'clinchacc',
'clinchatt',
'postcomp_clinchacc_perc',
'precomp_clinchacc_perc',
'postcomp_clinchacc_perc3',
'precomp_clinchacc_perc3',
'postcomp_clinchacc_perc5',
'precomp_clinchacc_perc5',
'groundacc',
'groundatt',
'postcomp_groundacc_perc',
'precomp_groundacc_perc',
'postcomp_groundacc_perc3',
'precomp_groundacc_perc3',
'postcomp_groundacc_perc5',
'precomp_groundacc_perc5',
'postcomp_winsum',
'precomp_winsum',
'postcomp_boutcount',
'precomp_boutcount',
'postcomp_winavg',
'precomp_winavg',
'postcomp_winsum3',
'precomp_winsum3',
'postcomp_winavg3',
'precomp_winavg3',
'postcomp_winsum5',
'precomp_winsum5',
'postcomp_winavg5',
'precomp_winavg5',
'postcomp_losssum',
'precomp_losssum',
'postcomp_lossavg',
'precomp_lossavg',
'postcomp_losssum3',
'precomp_losssum3',
'postcomp_lossavg3',
'precomp_lossavg3',
'postcomp_losssum5',
'precomp_losssum5',
'postcomp_lossavg5',
'precomp_lossavg5',
'postcomp_kosum',
'precomp_kosum',
'postcomp_koavg',
'precomp_koavg',
'postcomp_kosum3',
'precomp_kosum3',
'postcomp_koavg3',
'precomp_koavg3',
'postcomp_kosum5',
'precomp_kosum5',
'postcomp_koavg5',
'precomp_koavg5',
'postcomp_kodsum',
'precomp_kodsum',
'postcomp_kodavg',
'precomp_kodavg',
'postcomp_kodsum3',
'precomp_kodsum3',
'postcomp_kodavg3',
'precomp_kodavg3',
'postcomp_kodsum5',
'precomp_kodsum5',
'postcomp_kodavg5',
'precomp_kodavg5',
'postcomp_subwsum',
'precomp_subwsum',
'postcomp_subwavg',
'precomp_subwavg',
'postcomp_subwsum3',
'precomp_subwsum3',
'postcomp_subwavg3',
'precomp_subwavg3',
'postcomp_subwsum5',
'precomp_subsum5',
'postcomp_subwavg5',
'precomp_subwavg5',
'postcomp_subwdsum',
'precomp_subwdsum',
'postcomp_subwdavg',
'precomp_subwdavg',
'postcomp_subwdsum3',
'precomp_subdsum3',
'postcomp_subwdavg3',
'precomp_subwdavg3',
'postcomp_subwdsum5',
'precomp_subwdsum5',
'postcomp_subwdavg5',
'precomp_subwdavg5',
'postcomp_udecsum',
'precomp_udecsum',
'postcomp_udecavg',
'precomp_udecavg',
'postcomp_udecsum3',
'precomp_udecsum3',
'postcomp_udecavg3',
'precomp_udecavg3',
'postcomp_udecsum5',
'precomp_udecsum5',
'postcomp_udecavg5',
'precomp_udecavg5',
'postcomp_udecdsum',
'precomp_udecdsum',
'postcomp_udecdavg',
'precomp_udecdavg',
'postcomp_udecdsum3',
'precomp_udecdsum3',
'postcomp_udecdavg3',
'precomp_udecdavg3',
'postcomp_udecdsum5',
'precomp_udecdsum5',
'postcomp_udecdavg5',
'precomp_udecdavg5',
'opp_DATE',
'opp_jbout',
'opp_jevent',
'opp_jfighter',
'opp_EVENT',
'opp_BOUT',
'opp_FIGHTER',
'opp_fight_time_minutes',
'opp_precomp_tot_time_in_cage',
'opp_postcomp_tot_time_in_cage',
'opp_age',
'opp_HEIGHT',
'opp_WEIGHT',
'opp_REACH',
'opp_DOB',
'opp_sex',
'opp_weightindex',
'opp_weight_stat',
'opp_weight_of_fight',
'opp_weight_avg3',
'opp_win',
'opp_loss',
'opp_ko',
'opp_kod',
'opp_subw',
'opp_subwd',
'opp_udec',
'opp_udecd',
'opp_subatt',
'opp_postcomp_tot_time_in_cage_3',
'opp_precomp_tot_time_in_cage_3',
'opp_postcomp_tot_time_in_cage_5',
'opp_precomp_tot_time_in_cage_5',
'opp_postcomp_sigstr_pm',
'opp_precomp_sigstr_pm',
'opp_postcomp_sigstr_pm3',
'opp_precomp_sigstr_pm3',
'opp_postcomp_sigstr_pm5',
'opp_precomp_sigstr_pm5',
'opp_postcomp_tdavg',
'opp_precomp_tdavg',
'opp_postcomp_tdavg3',
'opp_precomp_tdavg3',
'opp_postcomp_tdavg5',
'opp_precomp_tdavg5',
'opp_sigstrabs',
'opp_postcomp_sapm',
'opp_precomp_sapm',
'opp_precomp_sapm3',
'opp_postcomp_sapm3',
'opp_precomp_sapm5',
'opp_postcomp_sapm5',
'opp_postcomp_subavg',
'opp_precomp_subavg',
'opp_postcomp_subavg3',
'opp_precomp_subavg3',
'opp_postcomp_subavg5',
'opp_precomp_subavg5',
'opp_tdabs',
'opp_tdattfromopp',
'opp_postcomp_tddef',
'opp_precomp_tddef',
'opp_postcomp_tddef3',
'opp_precomp_tddef3',
'opp_postcomp_tddef5',
'opp_precomp_tddef5',
'opp_ostrabs',
'opp_postcomp_ostrabssum',
'opp_sigstracc',
'opp_sigstratt',
'opp_precomp_sigstraccsum',
'opp_postcomp_sigstraccsum',
'opp_postcomp_sigstr_perc',
'opp_precomp_sigstr_perc',
'opp_precomp_sigstr_perc3',
'opp_postcomp_sigstr_perc3',
'opp_precomp_sigstr_perc5',
'opp_postcomp_sigstr_perc5',
'opp_sigstrattfromopp',
'opp_postcomp_strdef',
'opp_precomp_strdef',
'opp_postcomp_strdef3',
'opp_precomp_strdef3',
'opp_postcomp_strdef5',
'opp_precomp_strdef5',
'opp_tdacc',
'opp_tdatt',
'opp_postcomp_tdacc_perc',
'opp_precomp_tdacc_perc',
'opp_postcomp_tdacc_perc3',
'opp_precomp_tdacc_perc3',
'opp_postcomp_tdacc_perc5',
'opp_precomp_tdacc_perc5',
'opp_totalacc',
'opp_totalatt',
'opp_postcomp_totalacc_perc',
'opp_precomp_totalacc_perc',
'opp_postcomp_totalacc_perc3',
'opp_precomp_totalacc_perc3',
'opp_postcomp_totalacc_perc5',
'opp_precomp_totalacc_perc5',
'opp_headacc',
'opp_headatt',
'opp_postcomp_headacc_perc',
'opp_precomp_headacc_perc',
'opp_postcomp_headacc_perc3',
'opp_precomp_headacc_perc3',
'opp_postcomp_headacc_perc5',
'opp_precomp_headacc_perc5',
'opp_bodyacc',
'opp_bodyatt',
'opp_postcomp_bodyacc_perc',
'opp_precomp_bodyacc_perc',
'opp_postcomp_bodyacc_perc3',
'opp_precomp_bodyacc_perc3',
'opp_postcomp_bodyacc_perc5',
'opp_precomp_bodyacc_perc5',
'opp_legacc',
'opp_legatt',
'opp_postcomp_legacc_perc',
'opp_precomp_legacc_perc',
'opp_postcomp_legacc_perc3',
'opp_precomp_legacc_perc3',
'opp_postcomp_legacc_perc5',
'opp_precomp_legacc_perc5',
'opp_distacc',
'opp_distatt',
'opp_postcomp_distacc_perc',
'opp_precomp_distacc_perc',
'opp_postcomp_distacc_perc3',
'opp_precomp_distacc_perc3',
'opp_postcomp_distacc_perc5',
'opp_precomp_distacc_perc5',
'opp_clinchacc',
'opp_clinchatt',
'opp_postcomp_clinchacc_perc',
'opp_precomp_clinchacc_perc',
'opp_postcomp_clinchacc_perc3',
'opp_precomp_clinchacc_perc3',
'opp_postcomp_clinchacc_perc5',
'opp_precomp_clinchacc_perc5',
'opp_groundacc',
'opp_groundatt',
'opp_postcomp_groundacc_perc',
'opp_precomp_groundacc_perc',
'opp_postcomp_groundacc_perc3',
'opp_precomp_groundacc_perc3',
'opp_postcomp_groundacc_perc5',
'opp_precomp_groundacc_perc5',
'opp_postcomp_winsum',
'opp_precomp_winsum',
'opp_postcomp_boutcount',
'opp_precomp_boutcount',
'opp_postcomp_winavg',
'opp_precomp_winavg',
'opp_postcomp_winsum3',
'opp_precomp_winsum3',
'opp_postcomp_winavg3',
'opp_precomp_winavg3',
'opp_postcomp_winsum5',
'opp_precomp_winsum5',
'opp_postcomp_winavg5',
'opp_precomp_winavg5',
'opp_postcomp_losssum',
'opp_precomp_losssum',
'opp_postcomp_lossavg',
'opp_precomp_lossavg',
'opp_postcomp_losssum3',
'opp_precomp_losssum3',
'opp_postcomp_lossavg3',
'opp_precomp_lossavg3',
'opp_postcomp_losssum5',
'opp_precomp_losssum5',
'opp_postcomp_lossavg5',
'opp_precomp_lossavg5',
'opp_postcomp_kosum',
'opp_precomp_kosum',
'opp_postcomp_koavg',
'opp_precomp_koavg',
'opp_postcomp_kosum3',
'opp_precomp_kosum3',
'opp_postcomp_koavg3',
'opp_precomp_koavg3',
'opp_postcomp_kosum5',
'opp_precomp_kosum5',
'opp_postcomp_koavg5',
'opp_precomp_koavg5',
'opp_postcomp_kodsum',
'opp_precomp_kodsum',
'opp_postcomp_kodavg',
'opp_precomp_kodavg',
'opp_postcomp_kodsum3',
'opp_precomp_kodsum3',
'opp_postcomp_kodavg3',
'opp_precomp_kodavg3',
'opp_postcomp_kodsum5',
'opp_precomp_kodsum5',
'opp_postcomp_kodavg5',
'opp_precomp_kodavg5',
'opp_postcomp_subwsum',
'opp_precomp_subwsum',
'opp_postcomp_subwavg',
'opp_precomp_suwbavg',
'opp_postcomp_subwsum3',
'opp_precomp_subwsum3',
'opp_postcomp_subwavg3',
'opp_precomp_subwavg3',
'opp_postcomp_subwsum5',
'opp_precomp_subwsum5',
'opp_postcomp_subwavg5',
'opp_precomp_subwavg5',
'opp_postcomp_subwdsum',
'opp_precomp_subwdsum',
'opp_postcomp_subwdavg',
'opp_precomp_subwdavg',
'opp_postcomp_subwdsum3',
'opp_precomp_subwdsum3',
'opp_postcomp_subwdavg3',
'opp_precomp_subwdavg3',
'opp_postcomp_subwdsum5',
'opp_precomp_subwdsum5',
'opp_postcomp_subwdavg5',
'opp_precomp_subwdavg5',
'opp_postcomp_udecsum',
'opp_precomp_udecsum',
'opp_postcomp_udecavg',
'opp_precomp_udecavg',
'opp_postcomp_udecsum3',
'opp_precomp_udecsum3',
'opp_postcomp_udecavg3',
'opp_precomp_udecavg3',
'opp_postcomp_udecsum5',
'opp_precomp_udecsum5',
'opp_postcomp_udecavg5',
'opp_precomp_udecavg5',
'opp_postcomp_udecdsum',
'opp_precomp_udecdsum',
'opp_postcomp_udecdavg',
'opp_precomp_udecdavg',
'opp_postcomp_udecdsum3',
'opp_precomp_udecdsum3',
'opp_postcomp_udecdavg3',
'opp_precomp_udecdavg3',
'opp_postcomp_udecdsum5',
'opp_precomp_udecdsum5',
'opp_postcomp_udecdavg5',
'opp_precomp_udecdavg5'
  UNION ALL
SELECT
-- SApM : SApM - Significant Strikes Absorbed per Minute
-- Post Competition
single_fighter_view.DATE,
    single_fighter_view.jbout,
    single_fighter_view.jevent,
    single_fighter_view.jfighter,
    single_fighter_view.EVENT,
    single_fighter_view.BOUT,
    single_fighter_view.FIGHTER,
    single_fighter_view.fight_time_minutes,
    single_fighter_view.precomp_tot_time_in_cage,
    single_fighter_view.postcomp_tot_time_in_cage,
    single_fighter_view.age,
    single_fighter_view.HEIGHT,
    single_fighter_view.WEIGHT,
    single_fighter_view.REACH,
    single_fighter_view.DOB,
    single_fighter_view.sex,
    single_fighter_view.weightindex,
    single_fighter_view.weight_stat,
    single_fighter_view.weight_of_fight,
    single_fighter_view.weight_avg3,
    single_fighter_view.win,
    single_fighter_view.loss,
    single_fighter_view.ko,
    single_fighter_view.kod,
    single_fighter_view.subw,
    single_fighter_view.subwd,
    single_fighter_view.udec,
    single_fighter_view.udecd,
    single_fighter_view.subatt,
    single_fighter_view.postcomp_tot_time_in_cage_3,
    single_fighter_view.precomp_tot_time_in_cage_3,
    single_fighter_view.postcomp_tot_time_in_cage_5,
    single_fighter_view.precomp_tot_time_in_cage_5,
    single_fighter_view.postcomp_sigstr_pm,
    single_fighter_view.precomp_sigstr_pm,
    single_fighter_view.postcomp_sigstr_pm3,
    single_fighter_view.precomp_sigstr_pm3,
    single_fighter_view.postcomp_sigstr_pm5,
    single_fighter_view.precomp_sigstr_pm5,
    single_fighter_view.postcomp_tdavg,
    single_fighter_view.precomp_tdavg,
    single_fighter_view.postcomp_tdavg3,
    single_fighter_view.precomp_tdavg3,
    single_fighter_view.postcomp_tdavg5,
    single_fighter_view.precomp_tdavg5,
    single_fighter_view.sigstrabs,
    single_fighter_view.postcomp_sapm,
    single_fighter_view.precomp_sapm,
    single_fighter_view.precomp_sapm3,
    single_fighter_view.postcomp_sapm3,
    single_fighter_view.precomp_sapm5,
    single_fighter_view.postcomp_sapm5,
    single_fighter_view.postcomp_subavg,
    single_fighter_view.precomp_subavg,
    single_fighter_view.postcomp_subavg3,
    single_fighter_view.precomp_subavg3,
    single_fighter_view.postcomp_subavg5,
    single_fighter_view.precomp_subavg5,
    single_fighter_view.tdabs,
    single_fighter_view.tdattfromopp,
    single_fighter_view.postcomp_tddef,
    single_fighter_view.precomp_tddef,
    single_fighter_view.postcomp_tddef3,
    single_fighter_view.precomp_tddef3,
    single_fighter_view.postcomp_tddef5,
    single_fighter_view.precomp_tddef5,
    single_fighter_view.ostrabs,
    single_fighter_view.postcomp_ostrabssum,
    single_fighter_view.sigstracc,
    single_fighter_view.sigstratt,
    single_fighter_view.precomp_sigstraccsum,
    single_fighter_view.postcomp_sigstraccsum,
    single_fighter_view.postcomp_sigstr_perc,
    single_fighter_view.precomp_sigstr_perc,
    single_fighter_view.precomp_sigstr_perc3,
    single_fighter_view.postcomp_sigstr_perc3,
    single_fighter_view.precomp_sigstr_perc5,
    single_fighter_view.postcomp_sigstr_perc5,
    single_fighter_view.sigstrattfromopp,
    single_fighter_view.postcomp_strdef,
    single_fighter_view.precomp_strdef,
    single_fighter_view.postcomp_strdef3,
    single_fighter_view.precomp_strdef3,
    single_fighter_view.postcomp_strdef5,
    single_fighter_view.precomp_strdef5,
    single_fighter_view.tdacc,
    single_fighter_view.tdatt,
    single_fighter_view.postcomp_tdacc_perc,
    single_fighter_view.precomp_tdacc_perc,
    single_fighter_view.postcomp_tdacc_perc3,
    single_fighter_view.precomp_tdacc_perc3,
    single_fighter_view.postcomp_tdacc_perc5,
    single_fighter_view.precomp_tdacc_perc5,
    single_fighter_view.totalacc,
    single_fighter_view.totalatt,
    single_fighter_view.postcomp_totalacc_perc,
    single_fighter_view.precomp_totalacc_perc,
    single_fighter_view.postcomp_totalacc_perc3,
    single_fighter_view.precomp_totalacc_perc3,
    single_fighter_view.postcomp_totalacc_perc5,
    single_fighter_view.precomp_totalacc_perc5,
    single_fighter_view.headacc,
    single_fighter_view.headatt,
    single_fighter_view.postcomp_headacc_perc,
    single_fighter_view.precomp_headacc_perc,
    single_fighter_view.postcomp_headacc_perc3,
    single_fighter_view.precomp_headacc_perc3,
    single_fighter_view.postcomp_headacc_perc5,
    single_fighter_view.precomp_headacc_perc5,
    single_fighter_view.bodyacc,
    single_fighter_view.bodyatt,
    single_fighter_view.postcomp_bodyacc_perc,
    single_fighter_view.precomp_bodyacc_perc,
    single_fighter_view.postcomp_bodyacc_perc3,
    single_fighter_view.precomp_bodyacc_perc3,
    single_fighter_view.postcomp_bodyacc_perc5,
    single_fighter_view.precomp_bodyacc_perc5,
    single_fighter_view.legacc,
    single_fighter_view.legatt,
    single_fighter_view.postcomp_legacc_perc,
    single_fighter_view.precomp_legacc_perc,
    single_fighter_view.postcomp_legacc_perc3,
    single_fighter_view.precomp_legacc_perc3,
    single_fighter_view.postcomp_legacc_perc5,
    single_fighter_view.precomp_legacc_perc5,
    single_fighter_view.distacc,
    single_fighter_view.distatt,
    single_fighter_view.postcomp_distacc_perc,
    single_fighter_view.precomp_distacc_perc,
    single_fighter_view.postcomp_distacc_perc3,
    single_fighter_view.precomp_distacc_perc3,
    single_fighter_view.postcomp_distacc_perc5,
    single_fighter_view.precomp_distacc_perc5,
    single_fighter_view.clinchacc,
    single_fighter_view.clinchatt,
    single_fighter_view.postcomp_clinchacc_perc,
    single_fighter_view.precomp_clinchacc_perc,
    single_fighter_view.postcomp_clinchacc_perc3,
    single_fighter_view.precomp_clinchacc_perc3,
    single_fighter_view.postcomp_clinchacc_perc5,
    single_fighter_view.precomp_clinchacc_perc5,
    single_fighter_view.groundacc,
    single_fighter_view.groundatt,
    single_fighter_view.postcomp_groundacc_perc,
    single_fighter_view.precomp_groundacc_perc,
    single_fighter_view.postcomp_groundacc_perc3,
    single_fighter_view.precomp_groundacc_perc3,
    single_fighter_view.postcomp_groundacc_perc5,
    single_fighter_view.precomp_groundacc_perc5,
    single_fighter_view.postcomp_winsum,
    single_fighter_view.precomp_winsum,
    single_fighter_view.postcomp_boutcount,
    single_fighter_view.precomp_boutcount,
    single_fighter_view.postcomp_winavg,
    single_fighter_view.precomp_winavg,
    single_fighter_view.postcomp_winsum3,
    single_fighter_view.precomp_winsum3,
    single_fighter_view.postcomp_winavg3,
    single_fighter_view.precomp_winavg3,
    single_fighter_view.postcomp_winsum5,
    single_fighter_view.precomp_winsum5,
    single_fighter_view.postcomp_winavg5,
    single_fighter_view.precomp_winavg5,
    single_fighter_view.postcomp_losssum,
    single_fighter_view.precomp_losssum,
    single_fighter_view.postcomp_lossavg,
    single_fighter_view.precomp_lossavg,
    single_fighter_view.postcomp_losssum3,
    single_fighter_view.precomp_losssum3,
    single_fighter_view.postcomp_lossavg3,
    single_fighter_view.precomp_lossavg3,
    single_fighter_view.postcomp_losssum5,
    single_fighter_view.precomp_losssum5,
    single_fighter_view.postcomp_lossavg5,
    single_fighter_view.precomp_lossavg5,
    single_fighter_view.postcomp_kosum,
    single_fighter_view.precomp_kosum,
    single_fighter_view.postcomp_koavg,
    single_fighter_view.precomp_koavg,
    single_fighter_view.postcomp_kosum3,
    single_fighter_view.precomp_kosum3,
    single_fighter_view.postcomp_koavg3,
    single_fighter_view.precomp_koavg3,
    single_fighter_view.postcomp_kosum5,
    single_fighter_view.precomp_kosum5,
    single_fighter_view.postcomp_koavg5,
    single_fighter_view.precomp_koavg5,
    single_fighter_view.postcomp_kodsum,
    single_fighter_view.precomp_kodsum,
    single_fighter_view.postcomp_kodavg,
    single_fighter_view.precomp_kodavg,
    single_fighter_view.postcomp_kodsum3,
    single_fighter_view.precomp_kodsum3,
    single_fighter_view.postcomp_kodavg3,
    single_fighter_view.precomp_kodavg3,
    single_fighter_view.postcomp_kodsum5,
    single_fighter_view.precomp_kodsum5,
    single_fighter_view.postcomp_kodavg5,
    single_fighter_view.precomp_kodavg5,
    single_fighter_view.postcomp_subwsum,
    single_fighter_view.precomp_subwsum,
    single_fighter_view.postcomp_subwavg,
    single_fighter_view.precomp_subwavg,
    single_fighter_view.postcomp_subwsum3,
    single_fighter_view.precomp_subwsum3,
    single_fighter_view.postcomp_subwavg3,
    single_fighter_view.precomp_subwavg3,
    single_fighter_view.postcomp_subwsum5,
    single_fighter_view.precomp_subwsum5,
    single_fighter_view.postcomp_subwavg5,
    single_fighter_view.precomp_subwavg5,
    single_fighter_view.postcomp_subwdsum,
    single_fighter_view.precomp_subwdsum,
    single_fighter_view.postcomp_subwdavg,
    single_fighter_view.precomp_subwdavg,
    single_fighter_view.postcomp_subwdsum3,
    single_fighter_view.precomp_subwdsum3,
    single_fighter_view.postcomp_subwdavg3,
    single_fighter_view.precomp_subwdavg3,
    single_fighter_view.postcomp_subwdsum5,
    single_fighter_view.precomp_subwdsum5,
    single_fighter_view.postcomp_subwdavg5,
    single_fighter_view.precomp_subwdavg5,
    single_fighter_view.postcomp_udecsum,
    single_fighter_view.precomp_udecsum,
    single_fighter_view.postcomp_udecavg,
    single_fighter_view.precomp_udecavg,
    single_fighter_view.postcomp_udecsum3,
    single_fighter_view.precomp_udecsum3,
    single_fighter_view.postcomp_udecavg3,
    single_fighter_view.precomp_udecavg3,
    single_fighter_view.postcomp_udecsum5,
    single_fighter_view.precomp_udecsum5,
    single_fighter_view.postcomp_udecavg5,
    single_fighter_view.precomp_udecavg5,
    single_fighter_view.postcomp_udecdsum,
    single_fighter_view.precomp_udecdsum,
    single_fighter_view.postcomp_udecdavg,
    single_fighter_view.precomp_udecdavg,
    single_fighter_view.postcomp_udecdsum3,
    single_fighter_view.precomp_udecdsum3,
    single_fighter_view.postcomp_udecdavg3,
    single_fighter_view.precomp_udecdavg3,
    single_fighter_view.postcomp_udecdsum5,
    single_fighter_view.precomp_udecdsum5,
    single_fighter_view.postcomp_udecdavg5,
    single_fighter_view.precomp_udecdavg5,
    single_opponent_view.DATE as opp_DATE,
    single_opponent_view.jbout as opp_jbout,
    single_opponent_view.jevent as opp_jevent,
    single_opponent_view.jfighter as opp_jfighter,
    single_opponent_view.EVENT as opp_EVENT,
    single_opponent_view.BOUT as opp_BOUT,
    single_opponent_view.FIGHTER as opp_FIGHTER,
    single_opponent_view.fight_time_minutes as opp_fight_time_minutes,
    single_opponent_view.precomp_tot_time_in_cage as opp_precomp_tot_time_in_cage,
    single_opponent_view.postcomp_tot_time_in_cage as opp_postcomp_tot_time_in_cage,
    single_opponent_view.age as opp_age,
    single_opponent_view.HEIGHT as opp_HEIGHT,
    single_opponent_view.WEIGHT as opp_WEIGHT,
    single_opponent_view.REACH as opp_REACH,
    single_opponent_view.DOB as opp_DOB,
    single_opponent_view.sex as opp_sex,
    single_opponent_view.weightindex as opp_weightindex,
    single_opponent_view.weight_stat as opp_weight_stat,
    single_opponent_view.weight_of_fight as opp_weight_of_fight,
    single_opponent_view.weight_avg3 as opp_weight_avg3,
    single_opponent_view.win as opp_win,
    single_opponent_view.loss as opp_loss,
    single_opponent_view.ko as opp_ko,
    single_opponent_view.kod as opp_kod,
    single_opponent_view.subw as opp_subw,
    single_opponent_view.subwd as opp_subwd,
    single_opponent_view.udec as opp_udec,
    single_opponent_view.udecd as opp_udec,
    single_opponent_view.subatt as opp_subatt,
    single_opponent_view.postcomp_tot_time_in_cage_3 as opp_postcomp_tot_time_in_cage_3,
    single_opponent_view.precomp_tot_time_in_cage_3 as opp_precomp_tot_time_in_cage_3,
    single_opponent_view.postcomp_tot_time_in_cage_5 as opp_postcomp_tot_time_in_cage_5,
    single_opponent_view.precomp_tot_time_in_cage_5 as opp_precomp_tot_time_in_cage_5,
    single_opponent_view.postcomp_sigstr_pm as opp_postcomp_sigstr_pm,
    single_opponent_view.precomp_sigstr_pm as opp_precomp_sigstr_pm,
    single_opponent_view.postcomp_sigstr_pm3 as opp_postcomp_sigstr_pm3,
    single_opponent_view.precomp_sigstr_pm3 as opp_precomp_sigstr_pm3,
    single_opponent_view.postcomp_sigstr_pm5 as opp_postcomp_sigstr_pm5,
    single_opponent_view.precomp_sigstr_pm5 as opp_precomp_sigstr_pm5,
    single_opponent_view.postcomp_tdavg as opp_postcomp_tdavg,
    single_opponent_view.precomp_tdavg as opp_precomp_tdavg,
    single_opponent_view.postcomp_tdavg3 as opp_postcomp_tdavg3,
    single_opponent_view.precomp_tdavg3 as opp_precomp_tdavg3,
    single_opponent_view.postcomp_tdavg5 as opp_postcomp_tdavg5,
    single_opponent_view.precomp_tdavg5 as opp_precomp_tdavg5,
    single_opponent_view.sigstrabs as opp_sigstrabs,
    single_opponent_view.postcomp_sapm as opp_postcomp_sapm,
    single_opponent_view.precomp_sapm as opp_precomp_sapm,
    single_opponent_view.precomp_sapm3 as opp_precomp_sapm3,
    single_opponent_view.postcomp_sapm3 as opp_postcomp_sapm3,
    single_opponent_view.precomp_sapm5 as opp_precomp_sapm5,
    single_opponent_view.postcomp_sapm5 as opp_postcomp_sapm5,
    single_opponent_view.postcomp_subavg as opp_postcomp_subavg,
    single_opponent_view.precomp_subavg as opp_precomp_subavg,
    single_opponent_view.postcomp_subavg3 as opp_postcomp_subavg3,
    single_opponent_view.precomp_subavg3 as opp_precomp_subavg3,
    single_opponent_view.postcomp_subavg5 as opp_postcomp_subavg5,
    single_opponent_view.precomp_subavg5 as opp_precomp_subavg5,
    single_opponent_view.tdabs as opp_tdabs,
    single_opponent_view.tdattfromopp as opp_tdattfromopp,
    single_opponent_view.postcomp_tddef as opp_postcomp_tddef,
    single_opponent_view.precomp_tddef as opp_precomp_tddef,
    single_opponent_view.postcomp_tddef3 as opp_postcomp_tddef3,
    single_opponent_view.precomp_tddef3 as opp_precomp_tddef3,
    single_opponent_view.postcomp_tddef5 as opp_postcomp_tddef5,
    single_opponent_view.precomp_tddef5 as opp_precomp_tddef5,
    single_opponent_view.ostrabs as opp_ostrabs,
    single_opponent_view.postcomp_ostrabssum as opp_postcomp_ostrabssum,
    single_opponent_view.sigstracc as opp_sigstracc,
    single_opponent_view.sigstratt as opp_sigstratt,
    single_opponent_view.precomp_sigstraccsum as opp_precomp_sigstraccsum,
    single_opponent_view.postcomp_sigstraccsum as opp_postcomp_sigstraccsum,
    single_opponent_view.postcomp_sigstr_perc as opp_postcomp_sigstr_perc,
    single_opponent_view.precomp_sigstr_perc as opp_precomp_sigstr_perc,
    single_opponent_view.precomp_sigstr_perc3 as opp_precomp_sigstr_perc3,
    single_opponent_view.postcomp_sigstr_perc3 as opp_postcomp_sigstr_perc3,
    single_opponent_view.precomp_sigstr_perc5 as opp_precomp_sigstr_perc5,
    single_opponent_view.postcomp_sigstr_perc5 as opp_postcomp_sigstr_perc5,
    single_opponent_view.sigstrattfromopp as opp_sigstrattfromopp,
    single_opponent_view.postcomp_strdef as opp_postcomp_strdef,
    single_opponent_view.precomp_strdef as opp_precomp_strdef,
    single_opponent_view.postcomp_strdef3 as opp_postcomp_strdef3,
    single_opponent_view.precomp_strdef3 as opp_precomp_strdef3,
    single_opponent_view.postcomp_strdef5 as opp_postcomp_strdef5,
    single_opponent_view.precomp_strdef5 as opp_precomp_strdef5,
    single_opponent_view.tdacc as opp_tdacc,
    single_opponent_view.tdatt as opp_tdatt,
    single_opponent_view.postcomp_tdacc_perc as opp_postcomp_tdacc_perc,
    single_opponent_view.precomp_tdacc_perc as opp_precomp_tdacc_perc,
    single_opponent_view.postcomp_tdacc_perc3 as opp_postcomp_tdacc_perc3,
    single_opponent_view.precomp_tdacc_perc3 as opp_precomp_tdacc_perc3,
    single_opponent_view.postcomp_tdacc_perc5 as opp_postcomp_tdacc_perc5,
    single_opponent_view.precomp_tdacc_perc5 as opp_precomp_tdacc_perc5,
    single_opponent_view.totalacc as opp_totalacc,
    single_opponent_view.totalatt as opp_totalatt,
    single_opponent_view.postcomp_totalacc_perc as opp_postcomp_totalacc_perc,
    single_opponent_view.precomp_totalacc_perc as opp_precomp_totalacc_perc,
    single_opponent_view.postcomp_totalacc_perc3 as opp_postcomp_totalacc_perc3,
    single_opponent_view.precomp_totalacc_perc3 as opp_precomp_totalacc_perc3,
    single_opponent_view.postcomp_totalacc_perc5 as opp_postcomp_totalacc_perc5,
    single_opponent_view.precomp_totalacc_perc5 as opp_precomp_totalacc_perc5,
    single_opponent_view.headacc as opp_headacc,
    single_opponent_view.headatt as opp_headatt,
    single_opponent_view.postcomp_headacc_perc as opp_postcomp_headacc_perc,
    single_opponent_view.precomp_headacc_perc as opp_precomp_headacc_perc,
    single_opponent_view.postcomp_headacc_perc3 as opp_postcomp_headacc_perc3,
    single_opponent_view.precomp_headacc_perc3 as opp_precomp_headacc_perc3,
    single_opponent_view.postcomp_headacc_perc5 as opp_postcomp_headacc_perc5,
    single_opponent_view.precomp_headacc_perc5 as opp_precomp_headacc_perc5,
    single_opponent_view.bodyacc as opp_bodyacc,
    single_opponent_view.bodyatt as opp_bodyatt,
    single_opponent_view.postcomp_bodyacc_perc as opp_postcomp_bodyacc_perc,
    single_opponent_view.precomp_bodyacc_perc as opp_precomp_bodyacc_perc,
    single_opponent_view.postcomp_bodyacc_perc3 as opp_postcomp_bodyacc_perc3,
    single_opponent_view.precomp_bodyacc_perc3 as opp_precomp_bodyacc_perc3,
    single_opponent_view.postcomp_bodyacc_perc5 as opp_postcomp_bodyacc_perc5,
    single_opponent_view.precomp_bodyacc_perc5 as opp_precomp_bodyacc_perc5,
    single_opponent_view.legacc as opp_legacc,
    single_opponent_view.legatt as opp_legatt,
    single_opponent_view.postcomp_legacc_perc as opp_postcomp_legacc_perc,
    single_opponent_view.precomp_legacc_perc as opp_precomp_legacc_perc,
    single_opponent_view.postcomp_legacc_perc3 as opp_postcomp_legacc_perc3,
    single_opponent_view.precomp_legacc_perc3 as opp_precomp_legacc_perc3,
    single_opponent_view.postcomp_legacc_perc5 as opp_postcomp_legacc_perc5,
    single_opponent_view.precomp_legacc_perc5 as opp_precomp_legacc_perc5,
    single_opponent_view.distacc as opp_distacc,
    single_opponent_view.distatt as opp_distatt,
    single_opponent_view.postcomp_distacc_perc as opp_postcomp_distacc_perc,
    single_opponent_view.precomp_distacc_perc as opp_precomp_distacc_perc,
    single_opponent_view.postcomp_distacc_perc3 as opp_postcomp_distacc_perc3,
    single_opponent_view.precomp_distacc_perc3 as opp_precomp_distacc_perc3,
    single_opponent_view.postcomp_distacc_perc5 as opp_postcomp_distacc_perc5,
    single_opponent_view.precomp_distacc_perc5 as opp_precomp_distacc_perc5,
    single_opponent_view.clinchacc as opp_clinchacc,
    single_opponent_view.clinchatt as opp_clinchatt,
    single_opponent_view.postcomp_clinchacc_perc as opp_postcomp_clinchacc_perc,
    single_opponent_view.precomp_clinchacc_perc as opp_precomp_clinchacc_perc,
    single_opponent_view.postcomp_clinchacc_perc3 as opp_postcomp_clinchacc_perc3,
    single_opponent_view.precomp_clinchacc_perc3 as opp_precomp_clinchacc_perc3,
    single_opponent_view.postcomp_clinchacc_perc5 as opp_postcomp_clinchacc_perc5,
    single_opponent_view.precomp_clinchacc_perc5 as opp_precomp_clinchacc_perc5,
    single_opponent_view.groundacc as opp_groundacc,
    single_opponent_view.groundatt as opp_groundatt,
    single_opponent_view.postcomp_groundacc_perc as opp_postcomp_groundacc_perc,
    single_opponent_view.precomp_groundacc_perc as opp_precomp_groundacc_perc,
    single_opponent_view.postcomp_groundacc_perc3 as opp_postcomp_groundacc_perc3,
    single_opponent_view.precomp_groundacc_perc3 as opp_precomp_groundacc_perc3,
    single_opponent_view.postcomp_groundacc_perc5 as opp_postcomp_groundacc_perc5,
    single_opponent_view.precomp_groundacc_perc5 as opp_precomp_groundacc_perc5,
    single_opponent_view.postcomp_winsum as opp_postcomp_winsum,
    single_opponent_view.precomp_winsum as opp_precomp_winsum,
    single_opponent_view.postcomp_boutcount as opp_postcomp_boutcount,
    single_opponent_view.precomp_boutcount as opp_precomp_boutcount,
    single_opponent_view.postcomp_winavg as opp_postcomp_winavg,
    single_opponent_view.precomp_winavg as opp_precomp_winavg,
    single_opponent_view.postcomp_winsum3 as opp_postcomp_winsum3,
    single_opponent_view.precomp_winsum3 as opp_precomp_winsum3,
    single_opponent_view.postcomp_winavg3 as opp_postcomp_winavg3,
    single_opponent_view.precomp_winavg3 as opp_precomp_winavg3,
    single_opponent_view.postcomp_winsum5 as opp_postcomp_winsum5,
    single_opponent_view.precomp_winsum5 as opp_precomp_winsum5,
    single_opponent_view.postcomp_winavg5 as opp_postcomp_winavg5,
    single_opponent_view.precomp_winavg5 as opp_precomp_winavg5,
    single_opponent_view.postcomp_losssum as opp_postcomp_losssum,
    single_opponent_view.precomp_losssum as opp_precomp_losssum,
    single_opponent_view.postcomp_lossavg as opp_postcomp_lossavg,
    single_opponent_view.precomp_lossavg as opp_precomp_lossavg,
    single_opponent_view.postcomp_losssum3 as opp_postcomp_losssum3,
    single_opponent_view.precomp_losssum3 as opp_precomp_losssum3,
    single_opponent_view.postcomp_lossavg3 as opp_postcomp_lossavg3,
    single_opponent_view.precomp_lossavg3 as opp_precomp_lossavg3,
    single_opponent_view.postcomp_losssum5 as opp_postcomp_losssum5,
    single_opponent_view.precomp_losssum5 as opp_precomp_losssum5,
    single_opponent_view.postcomp_lossavg5 as opp_postcomp_lossavg5,
    single_opponent_view.precomp_lossavg5 as opp_precomp_lossavg5,
    single_opponent_view.postcomp_kosum as opp_postcomp_kosum,
    single_opponent_view.precomp_kosum as opp_precomp_kosum,
    single_opponent_view.postcomp_koavg as opp_postcomp_koavg,
    single_opponent_view.precomp_koavg as opp_precomp_koavg,
    single_opponent_view.postcomp_kosum3 as opp_postcomp_kosum3,
    single_opponent_view.precomp_kosum3 as opp_precomp_kosum3,
    single_opponent_view.postcomp_koavg3 as opp_postcomp_koavg3,
    single_opponent_view.precomp_koavg3 as opp_precomp_koavg3,
    single_opponent_view.postcomp_kosum5 as opp_postcomp_kosum5,
    single_opponent_view.precomp_kosum5 as opp_precomp_kosum5,
    single_opponent_view.postcomp_koavg5 as opp_postcomp_koavg5,
    single_opponent_view.precomp_koavg5 as opp_precomp_koavg5,
    single_opponent_view.postcomp_kodsum as opp_postcomp_kodsum,
    single_opponent_view.precomp_kodsum as opp_precomp_kodsum,
    single_opponent_view.postcomp_kodavg as opp_postcomp_kodavg,
    single_opponent_view.precomp_kodavg as opp_precomp_kodavg,
    single_opponent_view.postcomp_kodsum3 as opp_postcomp_kodsum3,
    single_opponent_view.precomp_kodsum3 as opp_precomp_kodsum3,
    single_opponent_view.postcomp_kodavg3 as opp_postcomp_kodavg3,
    single_opponent_view.precomp_kodavg3 as opp_precomp_kodavg3,
    single_opponent_view.postcomp_kodsum5 as opp_postcomp_kodsum5,
    single_opponent_view.precomp_kodsum5 as opp_precomp_kodsum5,
    single_opponent_view.postcomp_kodavg5 as opp_postcomp_kodavg5,
    single_opponent_view.precomp_kodavg5 as opp_precomp_kodavg5,
    single_opponent_view.postcomp_subwsum as opp_postcomp_subwsum,
    single_opponent_view.precomp_subwsum as opp_precomp_subwsum,
    single_opponent_view.postcomp_subwavg as opp_postcomp_subwavg,
    single_opponent_view.precomp_subwavg as opp_precomp_subwavg,
    single_opponent_view.postcomp_subwsum3 as opp_postcomp_subwsum3,
    single_opponent_view.precomp_subwsum3 as opp_precomp_subwsum3,
    single_opponent_view.postcomp_subwavg3 as opp_postcomp_subwavg3,
    single_opponent_view.precomp_subwavg3 as opp_precomp_subwavg3,
    single_opponent_view.postcomp_subwsum5 as opp_postcomp_subwsum5,
    single_opponent_view.precomp_subwsum5 as opp_precomp_subsum5,
    single_opponent_view.postcomp_subwavg5 as opp_postcomp_subwavg5,
    single_opponent_view.precomp_subwavg5 as opp_precomp_subwavg5,
    single_opponent_view.postcomp_subwdsum as opp_postcomp_subwdsum,
    single_opponent_view.precomp_subwdsum as opp_precomp_subwdsum,
    single_opponent_view.postcomp_subwdavg as opp_postcomp_subwdavg,
    single_opponent_view.precomp_subwdavg as opp_precomp_subwdavg,
    single_opponent_view.postcomp_subwdsum3 as opp_postcomp_subwdsum3,
    single_opponent_view.precomp_subwdsum3 as opp_precomp_subwdsum3,
    single_opponent_view.postcomp_subwdavg3 as opp_postcomp_subwdavg3,
    single_opponent_view.precomp_subwdavg3 as opp_precomp_subwdavg3,
    single_opponent_view.postcomp_subwdsum5 as opp_postcomp_subwdsum5,
    single_opponent_view.precomp_subwdsum5 as opp_precomp_subwdsum5,
    single_opponent_view.postcomp_subwdavg5 as opp_postcomp_subwdavg5,
    single_opponent_view.precomp_subwdavg5 as opp_precomp_subwdavg5,
    single_opponent_view.postcomp_udecsum as opp_postcomp_udecsum,
    single_opponent_view.precomp_udecsum as opp_precomp_udecsum,
    single_opponent_view.postcomp_udecavg as opp_postcomp_udecavg,
    single_opponent_view.precomp_udecavg as opp_precomp_udecavg,
    single_opponent_view.postcomp_udecsum3 as opp_postcomp_udecsum3,
    single_opponent_view.precomp_udecsum3 as opp_precomp_udecsum3,
    single_opponent_view.postcomp_udecavg3 as opp_postcomp_udecavg3,
    single_opponent_view.precomp_udecavg3 as opp_precomp_udecavg3,
    single_opponent_view.postcomp_udecsum5 as opp_postcomp_udecsum5,
    single_opponent_view.precomp_udecsum5 as opp_precomp_udecsum5,
    single_opponent_view.postcomp_udecavg5 as opp_postcomp_udecavg5,
    single_opponent_view.precomp_udecavg5 as opp_precomp_udecavg5,
    single_opponent_view.postcomp_udecdsum as opp_postcomp_udecdsum,
    single_opponent_view.precomp_udecdsum as opp_precomp_udecdsum,
    single_opponent_view.postcomp_udecdavg as opp_postcomp_udecdavg,
    single_opponent_view.precomp_udecdavg as opp_precomp_udecdavg,
    single_opponent_view.postcomp_udecdsum3 as opp_postcomp_udecdsum3,
    single_opponent_view.precomp_udecdsum3 as opp_precomp_udecdsum3,
    single_opponent_view.postcomp_udecdavg3 as opp_postcomp_udecdavg3,
    single_opponent_view.precomp_udecdavg3 as opp_precomp_udecdavg3,
    single_opponent_view.postcomp_udecdsum5 as opp_postcomp_udecdsum5,
    single_opponent_view.precomp_udecdsum5 as opp_precomp_udecdsum5,
    single_opponent_view.postcomp_udecdavg5 as opp_postcomp_udecdavg5,
    single_opponent_view.precomp_udecdavg5 as opp_precomp_udecdavg5
FROM single_fighter_view JOIN single_opponent_view
ON single_fighter_view.jbout = single_opponent_view.jbout AND
   single_fighter_view.jevent = single_opponent_view.jevent;


.output ufcdb.csv
select * from pre_final;
.output stdout
select count(*) from pre_final;

