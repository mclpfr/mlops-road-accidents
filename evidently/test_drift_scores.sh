#!/bin/bash

# Backup current data
BACKUP_DIR="/home/ubuntu/mlops-road-accidents/evidently/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp /home/ubuntu/mlops-road-accidents/evidently/current/current_data.csv "$BACKUP_DIR/"

echo "=== Testing different drift levels ==="

# Test each drift level
for drift_file in /home/ubuntu/mlops-road-accidents/evidently/test_data/test_data_drift_*.csv; do
    # Get drift level from filename
    drift_level=$(echo "$drift_file" | grep -oP 'drift_\K[0-9.]+')
    
    # Copy test file to current data
    cp "$drift_file" /home/ubuntu/mlops-road-accidents/evidently/current/current_data.csv
    
    # Wait a moment for the file to be fully written
    sleep 2
    
    # Get drift score
    echo -n "Drift level: ${drift_level} - "
    curl -s http://localhost:8001/drift_score | grep -v "^#"
    
    # Small delay between tests
    sleep 1
done

# Restore original data
cp "$BACKUP_DIR/current_data.csv" /home/ubuntu/mlops-road-accidents/evidently/current/

echo "=== Test completed. Original data restored. ==="
echo "Backup available in: $BACKUP_DIR"
