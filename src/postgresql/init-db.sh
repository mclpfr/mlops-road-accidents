#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE road_accidents;
    
    \c road_accidents;
    
    -- Table for accidents
    CREATE TABLE accidents (
        Num_Acc INT PRIMARY KEY,
        jour INT,
        mois INT,
        an INT,
        hrmn VARCHAR(10),
        lum INT,
        dep VARCHAR(5),
        com VARCHAR(5),
        agg INT,
        int INT,
        atm INT,
        col INT,
        adr VARCHAR(255),
        lat DOUBLE PRECISION,
        long DOUBLE PRECISION
    );
    
    GRANT ALL PRIVILEGES ON DATABASE road_accidents TO "$POSTGRES_USER";
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "$POSTGRES_USER";
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "$POSTGRES_USER";
EOSQL

echo "Database initialized successfully!" 
