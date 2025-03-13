import numpy as np

# Your list of lists
data = [[1, 2, 3], [2, 10, 20]]

# Convert the list of lists to a numpy array for easy cumulative sum operation
data_array = np.array(data)

# Calculate the cumulative sum along the rows (axis=0)
cumulative_sums = np.cumsum(data_array, axis=0)

# Convert back to a list of lists
cumulative_sums_list = cumulative_sums.tolist()

def get_csamples_from_dictstr(dstr):
    import re
    pattern = r"\'c_samples\':\s*(\d+)"
    match = re.search(pattern, dstr)

    assert match
    c_samples = int(match.group(1))  
    return c_samples

text = "{'experiment': 'A', 'discount': 0.9, 'horizon': 100, 'c_samples': 10, 'r_samples': 5}"

r = get_csamples_from_dictstr(text)
# print(r)

s = "Group {'experiment': 'tiger', 'discount': 0.9, 'horizon': 3, 'c_samples': 100, 'r_samples': 250}"
import ast
d = ast.literal_eval(s[6:])
print(d)