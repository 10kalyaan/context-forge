# import json
# from pathlib import Path
# from pprint import pprint

# DATA_PATH = Path("data/raw/locomo10.json")


# def main():
#     with open(DATA_PATH, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     print(f"Top-level type: {type(data)}")
#     print(f"Number of conversation records: {len(data)}")

#     first = data[0]
#     print("\nTop-level keys in first record:")
#     print(list(first.keys()))

#     print("\nFirst record preview:")
#     pprint(first)

#     # Try to inspect likely conversation/session fields
#     for key in first.keys():
#         value = first[key]
#         if isinstance(value, list):
#             print(f"\nKey '{key}' is a list with length {len(value)}")
#             if len(value) > 0:
#                 print(f"First item type: {type(value[0])}")
#                 print("First item preview:")
#                 pprint(value[0])
#         elif isinstance(value, dict):
#             print(f"\nKey '{key}' is a dict with keys: {list(value.keys())[:20]}")
#         else:
#             print(f"\nKey '{key}' -> {type(value)} -> {value}")


# if __name__ == "__main__":
#     main()


import json
from pathlib import Path
from pprint import pprint

DATA_PATH = Path("data/raw/locomo10.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

first = data[0]

print("sample_id:", first["sample_id"])
print("\nConversation keys:")
print(first["conversation"].keys())

print("\nsession_1_date_time:")
print(first["conversation"].get("session_1_date_time"))

session_1 = first["conversation"].get("session_1")
print("\nType of session_1:", type(session_1))

if isinstance(session_1, list):
    print("Length of session_1:", len(session_1))
    if len(session_1) > 0:
        print("\nFirst turn in session_1:")
        pprint(session_1[0])
else:
    print("\nsession_1 value:")
    pprint(session_1)
