import polars as pl

df = pl.read_csv("/data/datasets/habitat_recordings/data.csv")

task_to_split = "object_detection"

new_columns = df[task_to_split].unique().sort().drop_nulls().to_list()

print(new_columns)

df.with_columns(
    [
        pl.when(pl.col(task_to_split) == c)
        .then(pl.lit("yes"))
        .otherwise(pl.lit("no"))
        .alias(c)
        for c in new_columns
    ]
).drop(task_to_split).write_csv("/data/datasets/habitat_recordings/data_split.csv")
