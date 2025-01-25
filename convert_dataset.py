import csv
import json

# File paths
user_features_file = "twitter/user_features_train.txt"  # User features .txt file
json_file = "twitter/train_posts.json"  # Existing JSON file
output_file = "twitter/combined_traindata.json"  # Output file

# Step 1: Read the user features into a dictionary
user_features = {}
with open(user_features_file, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    # Normalize headers to strip any leading/trailing spaces
    reader.fieldnames = [header.strip() for header in reader.fieldnames]
    for row in reader:
        # Normalize row keys and values
        row = {key.strip(): value.strip() for key, value in row.items()}
        post_id = row["post_id"]
        user_features[post_id] = {
            "num_friends": int(row["num_friends"]),
            "num_followers": int(row["num_followers"]),
            "folfriend_ratio": float(row["folfriend_ratio"]),
            "times_listed": int(row["times_listed"]),
            "has_url": row["has_url"].lower() == "true",
            "is_verified": row["is_verified"].lower() == "true",
            "num_posts": int(row["num_posts"]),
        }

# Step 2: Load the JSON data
with open(json_file, "r", encoding="utf-8") as file:
    json_data = json.load(file)

# Step 3: Combine the data
for post in json_data:
    post_id = post["post_id"]
    if post_id in user_features:
        post.update(user_features[post_id])

# Step 4: Save the combined data to a new JSON file
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(json_data, file, indent=4, ensure_ascii=False)

print(f"Combined data has been saved to {output_file}.")

