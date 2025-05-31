import os

def test_check_accidents_csv():
    filename = 'data/raw/accidents_2023.csv'
    assert os.path.exists(filename)

