 
    # Question 1: Reverse List by N Elements

def reverse_by_n_elements(lst, n):
    result = []
    i = 0

    # Traverse the list in steps of n
    while i < len(lst):
        # Reverse the current group of n elements manually
        group = []
        for j in range(min(n, len(lst) - i)):  # Ensure we handle the last group correctly
            group.insert(0, lst[i + j])  # Insert each element at the start (reverse)

        result.extend(group)  # Append the reversed group to the result
        i += n  # Move to the next group

    return result
 # Question 2: Lists & Dictionaries

def group_by_length(strings):
    length_dict = {}

    # Traverse each string in the input list
    for string in strings:
        length = len(string)  # Calculate the length of the string

        # If the length is already a key, append the string to the list
        if length in length_dict:
            length_dict[length].append(string)
        # If the length is not a key, create a new list for that length
        else:
            length_dict[length] = [string]

    # Sort the dictionary by the keys (lengths) and return
    return dict(sorted(length_dict.items()))

   # Question 3: Flatten a Nested Dictionary
def flatten_dict(d, parent_key='', sep='.'):
    items = []

    # Traverse the dictionary
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k  # Concatenate parent and current key

        if isinstance(v, dict):
            # If the value is a dictionary, recursively flatten it
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # If the value is a list, iterate over its elements
            for i, item in enumerate(v):
                items.extend(flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
        else:
            # Base case: if it's neither a dict nor a list, add the flattened key-value pair
            items.append((new_key, v))

    return dict(items)

     # Question 4: Generate Unique Permutations
def unique_permutations(nums):
    def backtrack(path, used):
        # If the current permutation is complete, add it to the result
        if len(path) == len(nums):
            result.append(path[:])  # Add a copy of the current path (permutation)
            return

        for i in range(len(nums)):
            # Skip used elements or duplicates (i > 0 ensures it's not the first element)
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue

            # Include nums[i] in the current permutation and mark it as used
            path.append(nums[i])
            used[i] = True

            # Recursively generate the rest of the permutation
            backtrack(path, used)

            # Backtrack: remove the last element and mark it as unused
            path.pop()
            used[i] = False


# Question 5: Find All Dates in a Text
def find_all_dates(text):
    # Regular expressions for each date format
    regex_patterns = [
        r'\d{2}-\d{2}-\d{4}',  # dd-mm-yyyy
        r'\d{2}/\d{2}/\d{4}',  # mm/dd/yyyy
        r'\d{4}\.\d{2}\.\d{2}'  # yyyy.mm.dd
    ]

    # List to store all matched dates
    dates = []

    # Iterate over each regex pattern and find matches
    for pattern in regex_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)  # Add matches to the list of dates

    return dates

# Question 6: Decode Polyline, Convert to DataFrame with Distances
import polyline
import pandas as pd
import math

# Haversine function to calculate the distance between two points (in meters)
def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in meters
    R = 6371000
    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in meters
    return R * c

# Function to decode polyline and calculate distances
def decode_polyline_to_df(polyline_str):
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)

    # Convert to DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Initialize distance column
    df['distance'] = 0.0

    # Calculate the distance for each successive point using the Haversine formula
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, 'latitude'], df.loc[i - 1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']

        # Calculate the distance from the previous point
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)

    return df

# Question 7: Matrix Rotation and Transformation
def rotate_matrix_90_clockwise(matrix):
    # Transpose the matrix (swap rows and columns)
    transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    # Reverse each row to rotate by 90 degrees clockwise
    rotated = [row[::-1] for row in transposed]

    return rotated

def transform_matrix(matrix):
    n = len(matrix)
    transformed = [[0] * n for _ in range(n)]

    # Calculate the sum of all elements in the row and column, excluding the element itself
    for i in range(n):
        for j in range(n):
            row_sum = sum(matrix[i])  # Sum of the current row
            col_sum = sum(matrix[k][j] for k in range(n))  # Sum of the current column
            transformed[i][j] = row_sum + col_sum - 2 * matrix[i][j]  # Exclude the element itself
    return transformed

def rotate_and_transform(matrix):
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = rotate_matrix_90_clockwise(matrix)

    # Step 2: Transform the rotated matrix
    final_matrix = transform_matrix(rotated_matrix)

    return final_matrix


    