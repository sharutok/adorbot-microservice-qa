import json
import csv

_data=[]
with open("Step2.jsonl") as f:
    data = [json.loads(line) for line in f]
    _data.append(data)

with open("output.json","w")as file:
    file.write(json.dumps(_data[0]))


# for obj in _data[0]:
#     print(obj["custom_id"])
#     print(obj["response"]["body"]["data"][0]["embedding"])

with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(["Custom ID", "Embedding"])

    # Write the data rows
    for obj in _data[0]:
        custom_id = obj["custom_id"]
        embedding = obj["response"]["body"]["data"][0]["embedding"]
        # Convert embedding list to string for writing
        writer.writerow([custom_id, embedding])

print("Data written to output.csv")
