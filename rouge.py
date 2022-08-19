import csv
from torchmetrics.text.rouge import ROUGEScore

rouge = ROUGEScore()

out_data_file = open('rouge-stat-REDDIT_TIFU.csv', 'w')
header = ['R-1-reddit', 'R-1-mdt', 'R-2-reddit', 'R-2-mdt', 'R-L-reddit', 'R-L-mdt']
writer = csv.writer(out_data_file, delimiter='\t')
writer.writerow(header)

with open('./cnn_dm_today/shamane.reddit.only') as R_file, open('./cnn_dm_today/shamane.mdt.only') as M_file, open(
        './cnn_dm_today/shamane.target') as T_file:
    for R, M, T in zip(R_file, M_file, T_file):
        R = " ".join(R.split())
        M = " ".join(M.split())
        T = " ".join(T.split())

        mdt_rouge = rouge(M, T)
        reddit_rouge = rouge(R, T)

        line = [reddit_rouge['rouge1_fmeasure'], mdt_rouge['rouge1_fmeasure'], reddit_rouge['rouge2_fmeasure'],
                mdt_rouge['rouge2_fmeasure'], reddit_rouge['rougeL_fmeasure'], mdt_rouge['rougeL_fmeasure']]
        
        line_numpy = [float(T.numpy()) for T in line]
        writer.writerow(line_numpy)

