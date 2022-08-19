import csv
from torchmetrics.text.rouge import ROUGEScore

rouge = ROUGEScore()

out_data_file = open('TABLE-2.csv', 'w')
header = ['R-1-notoken', 'R-1-title', 'R-1-summary', 'R-2-notoken', 'R-2-title',
          'R-2-summary', 'R-L-notoken', 'R-L-title', 'R-L-summary']
writer = csv.writer(out_data_file, delimiter='\t')
writer.writerow(header)

with open('./cnn_dm_today/shamane.tokenless') as notoken_file, open(
        './cnn_dm_today/shamane.tokentitle') as tokentitle_file, open(
    './cnn_dm_today/hypo.tifu.final.bartssl.SUMMARY') as tokensummary_file, open(
    './cnn_dm_today/shamane.target') as T_file:
    for notoken, tokentitle, tokensummary, T in zip(notoken_file, tokentitle_file, tokensummary_file, T_file):
        no_token = " ".join(notoken.split())
        token_title = " ".join(tokentitle.split())
        token_summary = " ".join(tokensummary.split())
        T = " ".join(T.split())

        notoken_rouge = rouge(no_token, T)
        tokentitle_rouge = rouge(token_title, T)
        tokensummary_rouge = rouge(token_summary, T)

        line = [notoken_rouge['rouge1_fmeasure'], tokentitle_rouge['rouge1_fmeasure'],tokensummary_rouge['rouge1_fmeasure'],notoken_rouge['rouge2_fmeasure'], tokentitle_rouge['rouge2_fmeasure'],
                tokensummary_rouge['rouge2_fmeasure'],notoken_rouge['rougeL_fmeasure'], tokentitle_rouge['rougeL_fmeasure'],
                tokensummary_rouge['rougeL_fmeasure'],
                ]

        line_numpy = [float(T.numpy()) for T in line]
        writer.writerow(line_numpy)
