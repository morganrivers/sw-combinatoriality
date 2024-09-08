"""
This python script generates a dialogue using the augmented whale csv data. It does so by first labelling the
whale by its number ("Whale" column) and the words that it says. When another whale speaks, the constructedText
column is concatenated until that whale is interrupted. New conversations, defined by having a different "filename"
are separated and labelled by their filename.
"""
import pandas as pd

# Load the data
data = pd.read_csv('../data/sperm-whale-dialogues_augmented.csv')

# Initialize an empty list to hold the dialogues
dialogues = []

# Iterate over the rows in the dataframe
for i, row in data.iterrows():
    # If it's the start of a new conversation, add a new dialogue
    if i == 0 or row['File'] != data.loc[i - 1, 'File']:
        dialogues.append({
            'file': row['File'],
            'dialogue': []
        })

    # Add the whale's dialogue to the current conversation
    dialogues[-1]['dialogue'].append({
        'whale': row['Whale'],
        'text': row['ConstructedString']
    })

# # Print the dialogues
# for dialogue in dialogues:
#     print(f"File: {dialogue['file']}")
#     previous_whale_name = "" # empty as there was no previous, it's the start of a conversation
#     what_last_whale_said = ""
#     for line in dialogue['dialogue']:
#         if line['whale'] == previous_whale_name:
#             what_last_whale_said = what_last_whale_said + " " + line['text']
#         else:
#             previous_whale_name = line["whale"]
#             if not previous_whale_name == "" :
#                 # print what the last whale said if not a new conversation
#                 print(what_last_whale_said + ".") # end what it said with a period as it's the end of a sentence.
#             # new whale is speaking
#             what_last_whale_said = f"Whale {line['whale']}: {line['text']}"

# Open the output file
with open('../data/whale_dialogues.txt', 'w') as f:

    # Print the dialogues
    for dialogue in dialogues:
        # Initialize variables
        previous_whale_name = ""  # empty as there is no previous, it's the start of a conversation
        what_last_whale_said = ""

        # Write filename right above the start of the conversation to the file
        f.write(f"\nFile: {dialogue['file']}\n")

        for line in dialogue['dialogue']:
            if line['whale'] == previous_whale_name:
                what_last_whale_said = what_last_whale_said + " " + line['text']
            else:
                if not previous_whale_name == "":
                    # Write what the last whale said if not a new conversation to the file
                    f.write(what_last_whale_said + ".\n")  # end what it said with a period as it's the end of a sentence.

                previous_whale_name = line["whale"]
                # New whale is speaking
                what_last_whale_said = f"Whale {line['whale']}: {line['text']}"

        # Remember to write the last whale's sayings in each dialogue to the file
        f.write(what_last_whale_said + ".\n")
