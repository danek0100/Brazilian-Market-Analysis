from investiny import historical_data
from time import sleep
import csv

ids = []
data = []


with open('../resource/brazil_ids.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    #with open('../resource/data.csv', 'w', newline='') as csvfile:
     #   fieldnames = ['first_name', 'last_name']
      #  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
       # writer.writeheader()

    for row in reader:
        try:
            data.append(historical_data(investing_id=row['id'], from_date="01/01/2017", to_date="01/01/2018"))
            ids.append(row['id'])
            sleep(0.5)
        except Exception:
            continue


print(data)
