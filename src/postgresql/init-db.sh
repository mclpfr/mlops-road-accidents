#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE road_accidents' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'road_accidents')\gexec
    
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
    
    -- Table for best model metrics
    CREATE TABLE best_model_metrics (
        id SERIAL PRIMARY KEY,
        run_id VARCHAR(255),
        run_date TIMESTAMP,
        model_name VARCHAR(255),
        accuracy FLOAT,
        precision_macro_avg FLOAT,
        recall_macro_avg FLOAT,
        f1_macro_avg FLOAT,
        model_version INT,
        model_stage VARCHAR(50),
        year VARCHAR(4)
    );
    
    GRANT ALL PRIVILEGES ON DATABASE road_accidents TO "$POSTGRES_USER";
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "$POSTGRES_USER";
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "$POSTGRES_USER";
EOSQL

echo "Database initialized successfully!" 
