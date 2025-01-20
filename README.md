# Initial setup

Use Python3.11 or higher, with pip and venv installed.

```bash
poetry install --all-extras
```

When running locally, add a `.env` file with the following keys (fill in the values):
```
DATABASE_URL=
PGDATABASE=
PGHOST=
PGPORT=
PGUSER=
PGPASSWORD=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

## To get a shell with dependencies available

```bash
poetry shell
```

# Data import pipeline

1. Put new CSV files in the format of `book_name,parsha_name,dvar_torah_id,passage_id,passage_content` in the `data/raw_input_csv` directory.
2. Run the following to create an enriched version with translation, keywords and summary:
```bash
python data/preprocessing/enrich_csv_with_translation_and_keywords.py data/raw_input_csv/{yourfile} data/enriched/{yourfile} 10
```
3. Run the following to create a unified CSV file with all the enriched data:
```bash
python data/preprocessing/combine_enriched_csvs.py data/enriched/ data/dataset.csv
```

4. Run the following to create the guider text file:
```bash
python data/preprocessing/csv_to_text.py data/dataset.csv guider/dataset.txt
```


# Runtime

To run the server in dev mode:
```bash
hypercorn chassidic-ai.serving.server:app --reload
```

To run the server in prod mode:
```bash
hypercorn chassidic-ai.serving.server:app
```


For the new pipeline, run `run_retrieval.py`.
Results are stored under `run/`

For the old pipeline, run `app.py` and go to the shown url in your browser.
Results are stored under `data/answers/`
 
 # Data migration

 ## SQL data source (chassidic-sentences)

 Due to the size and encoding of the data, parsing the file directly or export import via a CSV/JSON (COPY in Postgres) did not work properly. Instead, data was imported into a docker container running Posgres, and then the Relational Migration Tool was used to migrate the data to Atlas. 

 Running the docker container:
 ```bash
 docker run --name chassidus-import -e POSTGRES_USER=root -e POSGRES_PASSWORD=rpwd -e POSTGRES_DB=db -p 5432:5432 -d postgres
 ```

 Copying the dump to the container (if running the import from within the container):
 ```bash
 docker cp ~/Downloads/chassidus_sentences.sql chassidus-import:/import.sql
 ```

 Importing into the container database (this is from the host machine):
```bash
docker exec -i chassidus-import psql -U root -d chassidus < ~/Downloads/chassidus_sentences.sql
```
Otherwise, if running from within the container:
```bash
# Log into the container
docker exec -it chassidus-import bash
# Import the dump
psql -U root -d chassidus < /import.sql
```

Download and run the relational migrator tool from https://www.mongodb.com/products/tools/relational-migrator

Connect to the source database (Postgres, db name `chassidus`) and the target database (Atlas).

Create a migration task with the default mapping, verification, and run the migration.

