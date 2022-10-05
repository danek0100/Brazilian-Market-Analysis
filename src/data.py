from investiny import historical_data
from time import sleep
import csv

ids = []
data = []
temp_data = {}


#data.append(historical_data(investing_id=18604, from_date="01/01/2017", to_date="01/01/2018"))

#print(data)

with open('../resource/brazil_ids.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            temp_data = historical_data(investing_id=row['id'], from_date="01/01/2017", to_date="01/01/2018")
            data.append(temp_data)
            sleep(1)
        except Exception:
            continue

        ids.append(row['id'])
        with open('../resource/data/' + row['id'] + '.csv', mode='w', encoding='utf-8') as w_file:
            column_names = ["close", "volume"]
            file_writer = csv.DictWriter(w_file, delimiter=",",
                                         lineterminator="\r", fieldnames=column_names)
            file_writer.writeheader()
            for i in range(len(temp_data['close'])):
                file_writer.writerow({"close": str(temp_data['close'][i]), "volume": str(temp_data['volume'][i])})


print(data)
