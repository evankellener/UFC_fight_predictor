-- Fix Wonky Times, Create WeightClass tables, and merge back into Tott table


-- Step 1 Remove wonky rows from fight ufc_fight_results
DELETE FROM ufc_fight_results
WHERE NOT `TIME FORMAT` in ('3 Rnd (5-5-5)', '5 Rnd (5-5-5-5-5)', '2 Rnd (5-5)', '3 Rnd + OT (5-5-5-5)') AND Round <> '1';

-- Manually assign weight index and sex classification to each weight class

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

-- update fight_results where WEIGHCLASS in (Open Weight Bout or Catch Weight Bout)
-- set sex == sex