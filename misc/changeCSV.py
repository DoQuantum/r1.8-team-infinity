import csv
from datetime import datetime

def get_quarter_label(date_str):
    dt_obj = datetime.strptime(date_str, "%m/%d/%Y")
    quarter = (dt_obj.month - 1) // 3 + 1
    return f"{dt_obj.year}Q{quarter}"

input_file = "snp500_quarters.csv"
output_file = "tickers_unpivoted.csv"

with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write header
    writer.writerow(["time_period", "date", "company"])

    for row in reader:
        date = row[0]              # first column
        tickers = row[1:]          # all remaining columns

        # compute quarter label
        quarter_label = get_quarter_label(date)

        # one output row per ticker
        for t in tickers:
            if t.strip() != "":   # skip empty columns
                writer.writerow([quarter_label, date, t])
