import pytest
from DataCleaning import parse_genres, clean_text

def test_parse_genres_valid():
    #verify parsing
    input_str = "['Horror', 'Fantasy']"
    correct_string = ['Horror', 'Fantasy']
    assert parse_genres(input_str) == correct_string

def test_parse_genres_invalid():
    #if an incorrectly formatted string is passed into the parsing genres, it will show an empty list
    assert parse_genres("Not a list") == []

def test_clean_text_removes_special_characters():
    #verify cleaning is consistent
    uncleaned = "This is a summary! This needs to be cleaned??"
    cleaned = clean_text(uncleaned)
    assert "!" not in cleaned and "?" not in cleaned
    assert cleaned.lower().startswith("this is a summary")