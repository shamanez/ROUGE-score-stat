import csv
from torchmetrics.text.rouge import ROUGEScore

rouge = ROUGEScore()

out_data_file = open('TABLE-1.csv', 'w')
header = ['R-1-xsum', 'R-1-cnndm', 'R-1-reddit', 'R-1-redditcnn', 'R-1-mixnocon', 'R-1-mix', 'R-2-xsum', 'R-2-cnndm',
          'R-2-reddit', 'R-2-redditcnn', 'R-2-mixnocon', 'R-2-mix', 'R-L-xsum', 'R-L-cnndm', 'R-L-reddit',
          'R-L-redditcnn', 'R-L-mixnocon', 'R-L-mix', ]
writer = csv.writer(out_data_file, delimiter='\t')
writer.writerow(header)


with open('./cnn_dm_today/shamane.XSUM') as Xsum_file, open('./cnn_dm_today/shamane.CNNDM') as Cnndm_file, open(
        './cnn_dm_today/shamane.reddit.only') as Ronly_file, open(
    './cnn_dm_today/shamane.reddit.bartcnn') as R_onlycnn, open(
    './cnn_dm_today/shamane.noconbartssl') as Mixncon_file, open('./cnn_dm_today/shamane.mdt.only') as Mix_file, open(
    './cnn_dm_today/shamane.target') as T_file:
    for xs, cn, R, Rcnn, Mnocon, M, T in zip(Xsum_file, Cnndm_file, Ronly_file, R_onlycnn, Mixncon_file, Mix_file,
                                             T_file):
        xsum = " ".join(xs.split())
        cnndm = " ".join(cn.split())
        reddit = " ".join(R.split())
        redditcnn = " ".join(Rcnn.split())
        mixnocon = " ".join(Mnocon.split())
        mix = " ".join(M.split())
        T = " ".join(T.split())



        xsum_rouge = rouge(M, T)
        cnndm_rouge = rouge(M, T)
        reddit_rouge = rouge(M, T)
        redditcnn_rouge = rouge(M, T)
        mixnocon_rouge = rouge(M, T)
        mix_rouge = rouge(M, T)



        line = [xsum_rouge['rouge1_fmeasure'], cnndm_rouge['rouge1_fmeasure'], reddit_rouge['rouge1_fmeasure'],
                redditcnn_rouge['rouge1_fmeasure'], mixnocon_rouge['rouge1_fmeasure'], mix_rouge['rouge1_fmeasure'],
                xsum_rouge['rouge2_fmeasure'], cnndm_rouge['rouge2_fmeasure'], reddit_rouge['rouge2_fmeasure'],
                redditcnn_rouge['rouge2_fmeasure'], mixnocon_rouge['rouge2_fmeasure'], mix_rouge['rouge2_fmeasure'],
                xsum_rouge['rougeL_fmeasure'], cnndm_rouge['rougeL_fmeasure'], reddit_rouge['rougeL_fmeasure'],
                redditcnn_rouge['rougeL_fmeasure'], mixnocon_rouge['rougeL_fmeasure'], mix_rouge['rougeL_fmeasure']
                ]

        line_numpy = [float(T.numpy()) for T in line]
        writer.writerow(line_numpy)

