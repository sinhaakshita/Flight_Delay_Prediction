# Flight Delay Prediction

This project predicts whether a US flight will be delayed by more than 15 minutes using historical flight data. It is implemented as a Google Colab notebook and uses AWS S3/Athena for data access.

## Overview

The notebook loads a 100,000-row flight sample, prepares features that are available before departure, and trains gradient-boosted classifiers to identify likely delays.

The prediction target is:

```text
IS_DELAYED = 1 when DEP_DELAY > 15 minutes
```

## Data

- Source file: `flights_sample_100k.csv`
- Storage: Amazon S3 bucket `akshita-flight-delay-2026`
- Athena database: `flight_delay_db`
- Dataset size used: 100,000 rows and 32 columns

The notebook supports two ways of loading the same data:

1. Download the CSV from S3 with `boto3`.
2. Run a SQL query in Amazon Athena with `awswrangler` and load the result into pandas.

## Features

The base model uses only information intended to be known before departure:

- `AIRLINE`
- `ORIGIN`
- `DEST`
- `MONTH`
- `DAY_OF_WEEK`
- `DEP_HOUR`
- `DISTANCE`

`MONTH`, `DAY_OF_WEEK`, and `DEP_HOUR` are derived from the flight date and scheduled departure time. The notebook deliberately excludes post-departure fields such as arrival delay and taxi-out time to avoid target leakage.

It also experiments with historical delay-rate features for the airline, origin airport, and scheduled departure hour.

## Models and results

The data is split into 80% training and 20% test sets using a stratified split. Categorical columns are label-encoded, and class imbalance is addressed with `scale_pos_weight`.

| Model | Test accuracy | ROC-AUC |
| --- | ---: | ---: |
| XGBoost - base features | 0.65 | 0.660 |
| XGBoost - with historical delay rates | 0.65 | 0.663 |
| LightGBM - with historical delay rates | 0.64 | 0.666 |

The sample is imbalanced: approximately 17.25% of flights are labeled delayed. ROC-AUC is therefore more informative than accuracy alone.

## Requirements

Run the notebook in Google Colab or a Python environment with these packages:

```bash
pip install boto3 awswrangler pandas scikit-learn xgboost lightgbm
```

You also need AWS credentials with permission to read the S3 object and query Athena.

## Configuration

Store credentials securely rather than placing them in the notebook or repository. In Colab, the notebook reads the following values from `userdata`:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

Set the AWS region to `ap-south-1`, then configure the S3 bucket, CSV key, and Athena database as needed for your account.

## Run the project

1. Open the notebook in Google Colab.
2. Add your AWS credentials to Colab Secrets.
3. Install the required packages.
4. Initialize the S3 client or `boto3` session.
5. Load `flights_sample_100k.csv` from S3 or Athena.
6. Create `IS_DELAYED` and engineering features.
7. Train and evaluate XGBoost and LightGBM.

## Notes and next steps

- Historical delay-rate features should be computed from training data only before applying them to the test set; computing them across the full dataset can leak test-set information.
- Replace label encoding with target-safe encodings or native categorical handling where appropriate.
- Tune the classification threshold based on the cost of missed delays versus false alerts.
- Validate the model with a time-based split for a more realistic production evaluation.
