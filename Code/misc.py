# import os

# def find_max_word_count(root_dir):
#     max_word_count = 0
#     longest_file = None
    
#     # Walk through the root directory and all subdirectories
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename.endswith('.txt'):
#                 file_path = os.path.join(dirpath, filename)
#                 # Read the file and count the number of words
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     word_count = len(f.read().split())
#                 # Update max_word_count and longest_file if this file has more words
#                 if word_count > max_word_count:
#                     max_word_count = word_count
#                     longest_file = file_path
                    
#     return max_word_count, longest_file

# # Set the path to your directory of directories
# root_directory = 'Data/Opinions'
# max_word_count, longest_file = find_max_word_count(root_directory)
# print(f"The longest .txt file by word count is: {longest_file}")
# print(f"Maximum word count: {max_word_count}")


import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# Initialize the tokenizer
model_name = 'google/bigbird-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def count_tokens(directory):
    token_counts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Tokenize the text to count tokens
            tokens = tokenizer(text, truncation=False)
            num_tokens = len(tokens['input_ids'])
            token_counts.append(num_tokens)

    return token_counts

# Count tokens in each document
directory = "Data/Amici"
token_counts = count_tokens(directory)

# Plot histogram of token counts
plt.figure(figsize=(10, 6))
plt.hist(token_counts, bins=30, color='mediumseagreen', edgecolor='black')
plt.axvline(x=4096, color='black', linestyle='--', label='4096 Token Limit')
plt.xlabel("Number of Tokens")
plt.ylabel("Number of Documents")
plt.title("Histogram of Amicus Curiae Token Counts")
plt.legend()
plt.show()
