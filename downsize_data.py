import pandas

## THIS FILE IS USED ONLY TO DOWNSIZE THE DATA ENOUGH TO BE UPLOADED TO GITHUB.
## ROWS ARE REMOVED AT RANDOM.
## IF YOU WISH TO SEE THE ORIGINAL DATASET, SEE LYKE ET AL. (2020).

input_file = "localdata/Clean_Quasar_Data.csv"
output_file = "data/Clean_Quasar_Data.csv"

df = pandas.read_csv(input_file)

num_rows_to_keep = len(df) // 4

# Takes random rows and samples them into new dataframe
df_reduced = df.sample(n=num_rows_to_keep, random_state=365)

df_reduced.to_csv(output_file, index=False)