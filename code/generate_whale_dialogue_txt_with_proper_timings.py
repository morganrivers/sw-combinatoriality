"""
This python script generates a dialogue using the augmented whale csv data. It does so by first labelling the
whale by its number ("Whale" column) and the words that it says. When another whale speaks, the constructedText
column is concatenated until another whale speaks or the conversation ends. New conversations, defined by having
a different "filename", are separated and labelled by their filename.

This also includes proper timings: printing the time of the new conversations, as well as reordering the display by
timestamp and printing when codas are spoken effectively simultaneously.

"""
import pandas as pd
import numpy as np
import os
from numpy import genfromtxt
import pickle
from IPython import embed
from scipy.io import loadmat

# Load the data
data = pd.read_csv('../data/sperm-whale-dialogues_augmented.csv')

# Parses, cleans and orders the annotations in the order of appearence of the clicks

# Initialize all the different conversations as empty arrays
annotations = {}
for i, row in data.iterrows():
    # If it's the start of a new conversation, add a new dialogue
    file_name = row['File'][:9]
    annotations[file_name] = []

# Then we go over the entries of the annotation file one by one (row wise) and append the annotation to the list corresponding to the file name
# The annotations are not ordered in time in the annotation file so it is a random order.

for i, row in data.iterrows():
    file_name = row['File'][:9]
    time_start = float(row['TsTo']) # start time of the first click of the coda seq

    # collect all files with that name
    temp = [] # List of times the clicks of the codas lie at - (t1-t1),(t2-t1),(t3-t1)...
    temp.append(0)
    j = 3
    while (float(row.iloc[j]) != 0 and j<31): # Until the value of the ICI is zero we continue to append (tn-t1)
        if float(row.iloc[j])> 0.0002: # To ignore mistakes in annotations
            temp.append(0+np.sum(row.iloc[3:(j+1)].astype(float))) # t3-t1 = (t3-t2)+(t2-t1) = ICI2+ICI1
        j = j+1
    annotations[file_name].append([temp, time_start, int(row['Whale']), row['ConstructedString'], 0]) # Coda seq, what_time, what_whale_number



# Annotation is now a dict of annotations but is still not ordered in time
# Convert the dict of lists into a list of numpy array
books = []
for key in annotations.keys():
    story = annotations[key] #each entity is a line of a coda


    book = np.zeros((len(story),44))
    words = [''] * len(story) # store all the "words" in the story.
    for j in range(len(story)):
        book[j,0] = story[j][2] # Who made the sound (ID number)
        book[j,1] = story[j][1] # What time did it start at
        book[j,2:len(story[j][0])+2] = story[j][0] # Coda sequence
        book[j,43] = story[j][-1] # Labels
        words[j] = story[j][3] # The string sequence corresponding to this word
    books.append((book,np.array(words),key))


# Sorts the annotation arrays of each audio file in order of time
edited = []
for i in range(len(books)):
    book = books[i][0]
    words = books[i][1]
    key = books[i][2]


    # both the words and the books are sorted by the time
    sorted_book = book[np.argsort(book[:, 1])]
    sorted_words = words[np.argsort(book[:, 1])]

    edited.append((sorted_book,sorted_words,key))

# CODA grouping parameters:
max_diff = 25 # Max time difference otherwise disconnect
max_diff = 15 # Max time difference otherwise disconnect
max_click_diff = 2 # Max  difference in the number of clicks, otherwise disconnect

# visualization parameters:
average_ICI = 0.2
duration_coda = 8*average_ICI # seconds for the vertical axis



previous_audiobook = ''
page_time_origin = -10000000000


newfile=0
starting_book =0
rootname='sw061b'
# duration_page = 60*3 # seconds for the horizontal axis


# Change this value depending on how many audiofiles you want to see plotted (one audio file may be split into several pages)
num_audiobooks = len(edited) # max value it can take = len(edited)

# Initialize an empty list to hold the dialogues
dialogues = []
# with open('../data/whale_dialogues_with_choruses.txt', 'w') as f:
for idx in range(num_audiobooks):
    book = edited[idx][0]
    words = edited[idx][1]
    name = edited[idx][2]
    rootname = name[0:6]
    # print("----------------------------------------------------------------")
    # print('idx: '+str(idx)+' File:'+name)
    # print('   number of codas='+str(book.shape[0]-1))
    # print('  '+ rootname +' previous=' + previous_audiobook)
    # print("----------------------------------------------------------------")
    # f.write(f"\nFile: {name}\n")


    dialogues.append({
        'file': name,
        'dialogue': []
    })

    # Function: parse de coda and get basic info
    def parseCoda(i):
        if i==-1:
            return [0,0,0,0,0,0]
        coda = book[i,:]
        word_string = words[i]
        whale_number = coda[0].astype(int)
        t_init = coda[1]-time_origin
        click_times = coda[2:42].astype(float).tolist()
        num_clicks = np.count_nonzero(click_times)+1 # sum 1 because the first click is always zero.
        click_times = click_times[0:num_clicks]
        label_ = coda[43].astype(int)
        # print(len([whale_number,t_init,word_string,num_clicks,click_times,label_]))
        return [whale_number,t_init,word_string,num_clicks,click_times,label_] #,average_power,click_power]

    # Function: returns the index for the next coda (after coda i) from the same whale.
    def getNextCoda(i, whale_number):
        found = -1
        while found==-1 and i<book.shape[0]-1:
            i=i+1
            [whale_number_i,t_init,word_string,num_clicks,click_times,label_] = parseCoda(i)
            if whale_number_i==whale_number:
                found = i
        return found

    # Function: returns the index for the previous coda (before coda i) from the same whale.
    def getPreviousCoda(i, whale_number):
        found = -1
        while found==-1 and i>0:
            i=i-1
            [whale_number_i,t_init,word_string,num_clicks,click_times,label_] = parseCoda(i)
            if whale_number_i==whale_number:
                found = i
        return found


    if rootname == previous_audiobook:
        page_time_origin = -10000000000
    else:
        time_origin = book[0,1]
        # print ('   time origin ='+str(time_origin))
        previous_audiobook = rootname
        newfile=1

    for i in range(book.shape[0]):

        [whale_number,t_init,word_string,num_clicks,click_times,label_curr] = parseCoda(i)
        i_next = getNextCoda(i, whale_number)
        [whale_number_next, t_init_next, word_string_next, num_clicks_next, click_times_next,label_next] = parseCoda(i_next)
        i_previous = getPreviousCoda(i, whale_number)
        [whale_number_previous, t_init_previous, word_string_previous, num_clicks_previous, click_times_previous,label_previous] = parseCoda(i_previous)
        dialogues[-1]['dialogue'].append({
            'whale': whale_number,
            'text': word_string,
            'timestamp': t_init
        })

        # if (i_previous > -1): # not the first entry in a book

        #     print("abs(prev_t_init - t_init)")
        #     print(abs(prev_t_init - t_init))

        #     if abs(prev_t_init - t_init) > 25:
        #         f.write(f"Long pause.\n")

        #     if abs(t_init - prev_t_init) < average_ICI/2 and prev_whale != whale_number:
        #         # simultaneous codas
        #         # print(f"In chorus, whales {whale_number} and {prev_whale}: {word_string}")
        #         if prev_word_string == word_string:
        #             f.write(f"In chorus, whales {prev_whale} and {whale_number}: {word_string}.\n")
        #         else:
        #             f.write(f"In chorus, whale {prev_whale}: {prev_word_string} and {whale_number}: {word_string}.\n")
        #     else:
        #         f.write(to_print_next)
        #         # write the previous entry normally if it's not a chorus
        #         to_print_next = f"Whale {whale_number}: {word_string}. (not chorus)\n"
        # else:
        #     # the first entry in a book
        #     to_print_next = f"Whale {whale_number}: {word_string}\n"


        # ### Plotting
        # print(f"Whale_number {whale_number}: {word_string}. ({t_init})")

        # if (i_next> -1) and (abs(len(click_times)-len(click_times_next))<=max_click_diff) and (t_init_next-t_init<max_diff):
        #     # this is indicating the lines plotted between a given phrase under these conditions.
        #     min_clicks = min(len(click_times), len(click_times_next))-1
        #     if abs(click_times_next[min_clicks]-click_times[min_clicks])<0.2:
        #         print(f"lines would be added for whale {whale_number}!")

        #     # for n in range(1,min(len(click_times), len(click_times_next))):
        #         # a=1

        #             # in this case, the click times for the last click of the coda were very close to one interclick interval
        #             # ax[splot].plot([t_init+click_times[n]-page_time_origin, t_init_next+click_times_next[n]-page_time_origin],
        #             #         [click_times[n], click_times_next[n]],
        #             #         '-',
        #             #       linewidth=15,
        #             #       color=colour_chart[whale_number])

# def save_the_dialogue_in_txt_format():
#     # Initialize an empty list to hold the dialogues
#     dialogues = []

#     # Iterate over the rows in the dataframe
#     for i, row in data.iterrows():
#         # If it's the start of a new conversation, add a new dialogue
#         if i == 0 or row['File'] != data.loc[i - 1, 'File']:
#             dialogues.append({
#                 'file': row['File'],
#                 'dialogue': []
#             })

#         # Add the whale's dialogue to the current conversation
#         dialogues[-1]['dialogue'].append({
#             'whale': row['Whale'],
#             'text': row['ConstructedString']
#         })


# In [12]: p = {}
#     ...: p['timestamp'] = 0
#     ...: for d in dialogues[0]['dialogue'][0:20]:
#     ...:     print(str(d) + str(d['timestamp'] - p['timestamp']))
#     ...:     p = d
#     ...:
# {'whale': 1, 'text': 'r3', 'timestamp': 0.0}0.0
# {'whale': 1, 'text': 'r5', 'timestamp': 2.28840000000001}2.28840000000001
# {'whale': 1, 'text': 'c3', 'timestamp': 5.710400000000007}3.421999999999997
# {'whale': 1, 'text': 'c3', 'timestamp': 9.550500000000014}3.8401000000000067
# {'whale': 1, 'text': 'c3', 'timestamp': 13.484500000000011}3.9339999999999975
# {'whale': 1, 'text': 'c3', 'timestamp': 16.750899999999987}3.266399999999976
# {'whale': 1, 'text': 'c3', 'timestamp': 19.819900000000004}3.069000000000017
# {'whale': 1, 'text': 'c3', 'timestamp': 23.358400000000003}3.538499999999999
# {'whale': 2, 'text': 'a3-', 'timestamp': 23.416600000000003}0.05819999999999936
# {'whale': 2, 'text': 'C4', 'timestamp': 26.77130000000001}3.3547000000000082
# {'whale': 1, 'text': 'c3', 'timestamp': 26.911000000000016}0.13970000000000482
# {'whale': 2, 'text': 'a4', 'timestamp': 29.980900000000005}3.06989999999999
# {'whale': 1, 'text': 'c3-', 'timestamp': 30.161800000000014}0.18090000000000828
# {'whale': 2, 'text': 'a4', 'timestamp': 34.9072}4.745399999999989
# {'whale': 2, 'text': 'a5', 'timestamp': 39.83970000000001}4.9325000000000045
# {'whale': 1, 'text': 'c4-', 'timestamp': 40.173800000000014}0.3341000000000065
# {'whale': 1, 'text': 'c4', 'timestamp': 44.3069}4.133099999999985
# {'whale': 2, 'text': 'a4-', 'timestamp': 44.544}0.2370999999999981
# {'whale': 2, 'text': 'a4', 'timestamp': 48.667199999999994}4.123199999999997
# {'whale': 1, 'text': 'c4-', 'timestamp': 48.768000000000015}0.10080000000002087

def print_chorus(chorus_whales_data, f):
    sorted_keys = sorted(chorus_whales_data)  # Sort the keys of the dictionary
    sorted_texts = [chorus_whales_data[key] for key in sorted_keys]  # Extract values in the sorted order of keys
    chorus_string = f"In chorus, whales {', '.join(map(str, sorted_keys))}: {' '.join(sorted_texts)}."
    f.write(chorus_string + "\n")


# save_the_dialogue_in_txt_format()
# print("dialogues[0] 10")
# print(dialogues[0])
# Open the output file
with open('../data/whale_dialogues.txt', 'w') as f:
    count = 0

    # Print the dialogues
    for dialogue in dialogues:
        # Initialize variables
        previous_whale_name = ""  # empty as there is no previous, it's the start of a conversation
        what_last_whale_said = ""
        what_last_whale_said_array = []
        last_timestamp = -np.inf
        chorus_texts = {}

        # Write filename right above the start of the conversation to the file
        f.write(f"\nFile: {dialogue['file']}\n")

        # Before starting the dialogue loop
        chorus_whales_data = {}
        chorus_text = ""
        last_chorus_time = -np.inf
        previously_in_chorus = False
        previous_whale_utterance = ""

        # Inside your dialogue loop
        for line in dialogue['dialogue']:
            print("line")
            print(line)
            print(f"COUNT: {count}")
            count = count + 1

            # if count >= 20:
            #     quit()

            this_timestamp = line['timestamp']
            assert last_timestamp <= this_timestamp

            # Check time difference and manage chorus
            time_diff = abs(this_timestamp - last_timestamp)
            print("time_diff")
            print(time_diff)

            if time_diff < 0.5 and previous_whale_name != line['whale']:
                # if previous_whale_name and what_last_whale_said: # if not on the first one
                if not previously_in_chorus:
                    # we don't need to repeat this one, it will be in the chorus, so go up to the penultamate entry
                    if len(what_last_whale_said_array) > 1:
                        f.write(f"Whale {previous_whale_name}: " + " ".join(what_last_whale_said_array[:-1]) + ".\n")
                what_last_whale_said = ""
                what_last_whale_said_array = []

                # Add current whale to chorus if not already in it
                if previous_whale_name not in chorus_whales_data.keys():
                    chorus_whales_data[previous_whale_name] = previous_whale_utterance
                if line['whale'] not in chorus_whales_data.keys():
                    chorus_whales_data[line['whale']] = line['text']

                last_chorus_time = this_timestamp
                previously_in_chorus = True
                print(f"IN CHORUS NOW, count {count}")

            else:
                # Output chorus if it exists and reset
                if previously_in_chorus: # if was chorus last time, and not this time.
                    print(f"IN CHORUS BEFORE, NO LONGER count {count}")
                    # chorus_string = f"In chorus, whales {', '.join(map(str, chorus_whales_data))}: {chorus_text}."
                    # f.write(chorus_string + "\n")
                    print_chorus(chorus_whales_data, f)
                    chorus_whales_data = {}
                    what_last_whale_said = f"Whale {line['whale']}: {line['text']}"
                    what_last_whale_said_array.append(line['text'])
                else:
                    # Continue with the regular logic
                    if line['whale'] == previous_whale_name:
                        what_last_whale_said += " " + line['text']
                        what_last_whale_said_array.append(line['text'])

                        print(f"Not chorus, same whale: THIS IS TRUE! for count {count}")
                        print(f"what_last whale said and this time: {what_last_whale_said}")
                    else: # not in a chorus, not previously in a chorus, and the whale is different.
                        if previous_whale_name and what_last_whale_said: # and, last whale said something, and not first entry
                            print(f"PREV AND WHAT LAST: THIS IS TRUE! for count {count}")
                            f.write(what_last_whale_said + ".\n")
                        what_last_whale_said = f"Whale {line['whale']}: {line['text']}"
                        what_last_whale_said_array.append(line['text'])
                previously_in_chorus = False
            previous_whale_name = line["whale"]
            previous_whale_utterance = line["text"]
            last_timestamp = this_timestamp
            if time_diff > 10:
                f.write(f"\nLong pause, {5*(time_diff//5)} seconds.\n")
        # After the loop, check if there's an unprocessed chorus
        if len(chorus_whales_data.keys()) == 0:
            print_chorus(chorus_whales_data, f)
        # Remember to write the last whale's sayings in each dialogue to the file
        f.write(what_last_whale_said + ".\n")
