import os

def test_check_prepared_accidents_csv():
    filename = 'data/processed/prepared_accidents_2023.csv'
    assert os.path.exists(filename)
