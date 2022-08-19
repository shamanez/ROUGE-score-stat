import csv
from torchmetrics.text.rouge import ROUGEScore

rouge = ROUGEScore()

out_data_file = open('TABLE-3.csv', 'w')
header = ['R-1-onlycnn', 'R-1-mixcnn', 'R-2-onlycnn', 'R-2-mixcnn', 'R-L-onlycnn', 'R-L-mixcnn']
writer = csv.writer(out_data_file, delimiter='\t')
writer.writerow(header)

with open('./cnn_dm_ori/shamane.cnndmonly') as cnndmonly_file, open(
        './cnn_dm_ori/shamane.summarizeme.cnndm') as cnndmsummarizedme_file, open(
    './cnn_dm_ori/cnndm.target') as T_file:
    for only, ours, T in zip(cnndmonly_file, cnndmsummarizedme_file, T_file):
        cnn_only = " ".join(only.split())
        ours = " ".join(ours.split())
        T = " ".join(T.split())

        cnnonly_rouge = rouge(cnn_only, T)
        ours_rouge = rouge(ours, T)

        line = [cnnonly_rouge['rouge1_fmeasure'], ours_rouge['rouge1_fmeasure'], cnnonly_rouge['rouge2_fmeasure'],
                ours_rouge['rouge2_fmeasure'], cnnonly_rouge['rougeL_fmeasure'], ours_rouge['rougeL_fmeasure']
                ]

        line_numpy = [float(T.numpy()) for T in line]
        writer.writerow(line_numpy)
