Fact checking

## Environment

```
pip install pytorch-lightning==1.9.5 pandas transformers pytorch-nlp scikit-learn
pip install readability-lxml  # Used to extract the title and content of a webpage.
pip install BeautifulSoup4    # Used for parsing HTML files.
pip install google-api-python-client # Used for Google search
pip install readability # Used for extracting title and main text of a html
pip install spacy       # Used for extracting entities from a text
```

## How to run the code

```
python training.py --min_epochs 5 --max_epochs 80 --batch_size 16 \
--save_dir /path/to/experiments/ \
--train_csv /path/to/train.csv \
--dev_csv /path/to/dev.csv \
--test_csv /path/to/test.csv
```

## Spacy Entities

- PERSON - People, including fictional.
- NORP - Nationalities or religious or political groups.
- FAC - Buildings, airports, highways, bridges, etc.
- ORG - Companies, agencies, institutions, etc.
- GPE - Countries, cities, states.
- LOC - Non-GPE locations, mountain ranges, bodies of water.
- PRODUCT - Objects, vehicles, foods, etc. (Not services.)
- EVENT - Named hurricanes, battles, wars, sports events, etc.
- WORK_OF_ART - Titles of books, songs, etc.
- LAW - Named documents made into laws.
- LANGUAGE - Any named language.
- DATE - Absolute or relative dates or periods.
- TIME - Times smaller than a day.
- PERCENT - Percentage, including "%".
- MONEY - Monetary values, including unit.
- QUANTITY - Measurements, as of weight or distance.
- ORDINAL - "first", "second", etc.
- CARDINAL - Numerals that do not fall under another type.
