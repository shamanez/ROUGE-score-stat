import csv

input_file = csv.DictReader(open("counselchat-data.csv"))

source_list = []

i=0

word_len=0

with open('./shamane.source','w') as src, open('./shamane.target', 'w') as fout:

        for row in input_file:
            summary = row['questionTitle'].split()
            summary = " ".join(summary)

            source = row['questionText'].split()
            source = " ".join(source)

            if source in source_list:
                continue

            source_list.append(source)

            if len(source.split()) < 50:
                continue


            word_len = word_len  + len(summary.split())
            fout.write(summary + '\n')
            src.write(source + '\n')

            i=i+1




print(word_len/i)


# avg number of words in a summary  - 10



