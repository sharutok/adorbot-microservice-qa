import pandas as pd
import json
import os
import uuid

# Load Excel file
excel_file = "URIBook2.xlsm"  # Replace with your Excel file path
sheet_name = "abc"  # Replace with the sheet name you want to convert

current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)
excel_file_path = os.path.join(current_directory,  "./URIBook1.xlsm")
print(excel_file_path)

# Read Excel sheet into a DataFrame
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Convert DataFrame to JSON
json_data = df.to_json(orient="records")
data=json.loads(json_data)

def _uuid():
    return str(uuid.uuid4())

_data=[]
for d in data:
    print(d)
    d["custom_id"] = _uuid()
    d["method"] = "POST"
    d["url"]= "/v1/embeddings"
    d["body"] = {
        "model": "text-embedding-3-small",
        "input": d["PDF NAME"],
        "encoding_format": "float",
    }
    del d["PDF NAME"]
    del d["Links"]

    _data.append(d)
json.dumps(_data)


json_file = "Step1.jsonl"  
with open(json_file, "w") as file:
    for dd in _data:
        file.write(json.dumps(dd)+'\n')
        # json.dump(_data, file, indent=4)

print(f"Excel data has been converted to JSON and saved to {json_file}.")
