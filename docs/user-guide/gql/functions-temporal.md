---
title: Temporal Functions
description: GQL date, time, datetime, duration and zoned temporal functions in Obrain.
tags:
  - gql
  - functions
  - temporal
---

# Temporal Functions

## Summary

| Function | Description |
|----------|-------------|
| `current_date()` | Today's date |
| `current_time()` | Current time |
| `now()` / `current_timestamp()` | Current datetime |
| `date()` / `date(str)` | Today / parse date |
| `time()` / `time(str)` | Current time / parse time |
| `datetime(str)` / `localdatetime()` | Parse or current datetime |
| `duration(str)` | Parse ISO 8601 duration |
| `year(val)` | Extract year |
| `month(val)` | Extract month |
| `day(val)` | Extract day |
| `hour(val)` | Extract hour |
| `minute(val)` | Extract minute |
| `second(val)` | Extract second |
| `toDate(expr)` | Convert to date |
| `toTime(expr)` | Convert to time |
| `toDatetime(expr)` | Convert to datetime |
| `toDuration(expr)` | Convert to duration |
| `toZonedDatetime(expr)` | Convert to zoned datetime |
| `toZonedTime(expr)` | Convert to zoned time |

## Current Date and Time

```sql
-- Today's date
RETURN current_date()       -- e.g., 2024-06-15

-- Current time
RETURN current_time()       -- e.g., 14:30:00

-- Current datetime (timestamp)
RETURN now()                -- e.g., 2024-06-15T14:30:00
RETURN current_timestamp()  -- alias for now()
```

## Constructors

Parse temporal values from strings:

```sql
-- Date
RETURN date('2024-01-15')

-- Time
RETURN time('14:30:00')

-- Datetime
RETURN datetime('2024-01-15T14:30:00')
RETURN localdatetime()  -- current local datetime

-- Duration (ISO 8601 format)
RETURN duration('P1Y')       -- 1 year
RETURN duration('P1Y2M')     -- 1 year, 2 months
RETURN duration('P1Y2M3D')   -- 1 year, 2 months, 3 days
RETURN duration('PT12H30M')  -- 12 hours, 30 minutes
RETURN duration('P1DT2H')    -- 1 day, 2 hours
```

## Typed Temporal Literals

Use typed literal syntax for inline temporal values:

```sql
-- Date literal
RETURN DATE '2024-01-15'

-- Time literal
RETURN TIME '14:30:00'

-- Datetime literal
RETURN DATETIME '2024-01-15T14:30:00Z'

-- Duration literal
RETURN DURATION 'P1Y2M3D'

-- Zoned datetime literal (with UTC offset)
RETURN ZONED DATETIME '2024-01-15T14:30:00+05:30'

-- Zoned time literal (with UTC offset)
RETURN ZONED TIME '14:30:00+05:30'
```

## Zoned Temporals

Zoned datetime and zoned time carry a fixed UTC offset:

```sql
-- Create zoned datetime from CAST
RETURN CAST('2024-01-15T14:30:00+05:30' AS ZONED DATETIME)

-- Create zoned time
RETURN CAST('14:30:00-04:00' AS ZONED TIME)

-- Conversion functions
RETURN toZonedDatetime('2024-01-15T14:30:00+05:30')
RETURN toZonedTime('14:30:00+05:30')

-- Practical: store with timezone info
MATCH (e:Event)
SET e.starts_at = ZONED DATETIME '2024-06-15T09:00:00+02:00'
```

## Component Extraction

Extract individual components from temporal values:

```sql
-- From a date
WITH DATE '2024-06-15' AS d
RETURN year(d), month(d), day(d)
-- 2024, 6, 15

-- From a time
WITH TIME '14:30:45' AS t
RETURN hour(t), minute(t), second(t)
-- 14, 30, 45

-- From a datetime
WITH DATETIME '2024-06-15T14:30:45Z' AS dt
RETURN year(dt), month(dt), day(dt),
       hour(dt), minute(dt), second(dt)
-- 2024, 6, 15, 14, 30, 45

-- Practical: group events by month
MATCH (e:Event)
RETURN year(e.created_at) AS yr,
       month(e.created_at) AS mo,
       count(*) AS event_count
ORDER BY yr, mo
```

## Type Conversion

Convert between temporal types:

```sql
-- String to temporal
RETURN toDate('2024-01-15')
RETURN toTime('14:30:00')
RETURN toDatetime('2024-01-15T14:30:00Z')
RETURN toDuration('P1Y2M')

-- CAST syntax (equivalent)
RETURN CAST('2024-01-15' AS DATE)
RETURN CAST('14:30:00' AS TIME)
RETURN CAST('2024-01-15T14:30:00Z' AS DATETIME)
RETURN CAST('P1Y2M' AS DURATION)
```

## Temporal Arithmetic

Add and subtract durations from temporal values:

```sql
-- Date + duration
RETURN DATE '2024-01-15' + DURATION 'P30D'
-- 2024-02-14

-- Datetime + duration
RETURN DATETIME '2024-01-15T14:30:00Z' + DURATION 'PT2H30M'

-- Practical: find events in the last 7 days
MATCH (e:Event)
WHERE e.created_at > now() - DURATION 'P7D'
RETURN e.title, e.created_at

-- Practical: set expiration date
MATCH (s:Subscription {plan: 'annual'})
SET s.expires_at = s.started_at + DURATION 'P1Y'
```
