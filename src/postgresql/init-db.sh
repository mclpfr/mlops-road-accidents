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
    
    -- Table for model metrics
    CREATE TABLE model_metrics (
        id SERIAL PRIMARY KEY,
        run_id VARCHAR(255),
        run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        model_name VARCHAR(255),
        accuracy DOUBLE PRECISION,
        precision_macro_avg DOUBLE PRECISION,
        recall_macro_avg DOUBLE PRECISION,
        f1_macro_avg DOUBLE PRECISION,
        model_version VARCHAR(255),
        year VARCHAR(4)
    );
    
    -- Access rights for PostgreSQL user
    GRANT ALL PRIVILEGES ON DATABASE road_accidents TO "$POSTGRES_USER";
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "$POSTGRES_USER";
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "$POSTGRES_USER";
EOSQL

echo "Database initialized successfully!" 