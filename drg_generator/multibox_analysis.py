from pathlib import Path

pmb4_train = 'data/data_split/gold4/en_train.txt'
pmb4_train_penman = 'dataud_boxer/data_split/gold4/gold_train.txt'
pmb4_dev = 'data/data_split/gold4/en_dev.txt'
pmb4_dev_penman = 'data/data_split/gold4/gold_dev.txt'
pmb4_test ='data/data_split/gold4/en_test.txt'
pmb4_test_penman = 'data/data_split/gold4/gold_test.txt'
pmb4_eval ='data/data_split/gold4/en_eval.txt'
pmb4_eval_penman = 'data/data_split/gold4/gold_eval.txt'


pmb5_train = 'data/data_split/gold5/en_train.txt'
pmb5_train_penman= 'data/gold5/gold_train.txt'
pmb5_dev ='data/data_split/gold5/en_dev.txt'
pmb5_dev_penman = 'data/data_split/gold5/gold_dev.txt'
pmb5_test ='data/data_split/gold4/en_test.txt'
pmb5_test_penman = 'data/data_split/gold5/gold_test.txt'
pmb5_testlong ='data/en_test_long.txt'
pmb5_testlong_penman = 'data/gold5/gold_test_long.txt'

def get_multibox(gold_path, gold_penman, box1, box2='b10'):
    multi_path =[]
    multi_penman =[]
    gold_path = Path(gold_path).read_text().strip().split('\n')
    gold_penman = Path(gold_penman).read_text().strip().split('\n\n')

    for path, penman in zip(gold_path, gold_penman):

        if box1 in penman and box2 not in penman:

            multi_path.append(path)
            multi_penman.append(penman)
    return multi_path, multi_penman

def get_pred(predpath, pred, multi_gold_path):
    pred_multi_penman =[]
    pred_path = Path(predpath).read_text().strip().split('\n')
    pred_penman = Path(pred).read_text().strip().split('\n\n')
    multibox_dic = {x:y for x, y in zip(pred_path, pred_penman)}
    for p in multi_gold_path:
        pred_penman = multibox_dic[p]
        pred_multi_penman.append(pred_penman)
    return pred_multi_penman

def output_pred(output_dir, penman_pred):
    with open(output_dir, 'w') as out:
        for p in penman_pred:
            out.write(f'{p}\n\n')


multi_path_dev4, multi_penman_dev4 = get_multibox(pmb4_dev, pmb4_dev_penman, 'b1')
multi_path_test4, multi_penman_test4 = get_multibox(pmb4_test, pmb4_test_penman, 'b1')
multi_path_eval4, multi_penman_eval4 = get_multibox(pmb4_eval, pmb4_eval_penman, 'b1')

# multi_path_dev5, multi_penman_dev5 = get_multibox(pmb5_dev, pmb5_dev_penman, 'b1')
# multi_path_test5, multi_penman_test5 = get_multibox(pmb5_test, pmb5_test_penman, 'b1')
# multi_path_testlong5, multi_penman_testlong5 = get_multibox(pmb5_testlong, pmb5_testlong_penman, 'b1')
multi_path4 = multi_path_dev4+multi_path_test4+multi_path_eval4
# multi_path5 = multi_path_dev5+multi_path_test5
multi_gold4 = multi_penman_dev4+multi_penman_test4+multi_penman_eval4
# multi_gold5 = multi_penman_dev5+multi_penman_test5

gold_dev_output_path = '/ud_boxer/output/output_multibox/4/gold.txt'
udboxer_dev ='data/seq2seqoutput/4/t5_dev_gold.txt'
udboxer_test = 'data/seq2seqoutput/4/t5_test_gold.txt'
udboxer_eval = 'data/seq2seqoutput/4/t5_test_gold.txt'
udboxer_multioutput ='data/output_multibox/4/t5.txt'
udboxer_dev_penman = get_pred(pmb4_dev, udboxer_dev, multi_path_dev4)
udboxer_test_penman = get_pred(pmb4_test, udboxer_test, multi_path_test4)
udboxer_eval_penman = get_pred(pmb4_eval, udboxer_eval, multi_path_eval4)

udboxer_penman = udboxer_dev_penman+udboxer_test_penman+udboxer_eval_penman
# udboxer_penman = udboxer_dev_penman+udboxer_test_penman
output_pred(udboxer_multioutput, udboxer_penman)
output_pred(gold_dev_output_path, multi_gold4)

# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np
# plt.style.use('seaborn')
# matplotlib.rcParams.update({'font.size': 9})
# # Data for all four groups
# data_group1 = statistics_train5
# data_group2 = statistics_dev5
# data_group3 = statistics_test5
# data_group4 = statistics_eval5
# f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
# # Extract keys and values for each group
# keys_group1, values_group1 = zip(*data_group1.items())
# keys_group2, values_group2 = zip(*data_group2.items())
# keys_group3, values_group3 = zip(*data_group3.items())
# keys_group4, values_group4 = zip(*data_group4.items())
#
# # Define the width#
# # def get_box(gold_path, gold_penman):
# #
# #     gold_path = Path(gold_path).read_text().strip().split('\n')
# #     gold_penman = Path(gold_penman).read_text().strip().split('\n\n')
# #
# #     statistics ={1:0, 2:0, 3:0, 4:0, '>4':0}
# #     for path, penman in zip(gold_path, gold_penman):
# #
# #         if 'b4' in penman:
# #             statistics['>4']+=1
# #         elif 'b3' in penman:
# #             statistics[4]+=1
# #         elif 'b2' in penman:
# #             statistics[3]+=1
# #         elif 'b1' in penman:
# #             statistics[2]+=1
# #         elif 'b0' in penman:
# #             statistics[1]+=1
# #         else:
# #             print(path)
# #             print('wtf')
# #
# #     return statistics
# #
# # statistics_train4 = get_box(pmb4_train, pmb4_train_penman)
# # statistics_dev4 = get_box(pmb4_dev, pmb4_dev_penman)
# # statistics_test4= get_box(pmb4_test, pmb4_test_penman)
# # statistics_eval4 = get_box(pmb4_eval, pmb4_eval_penman)
# #
# # statistics_dev5 = get_box(pmb5_dev, pmb5_dev_penman)
# # statistics_test5 =get_box(pmb5_test, pmb5_test_penman)
# # statistics_train5 = get_box(pmb5_train, pmb5_train_penman)
# # statistics_eval5 = get_box(pmb5_testlong, pmb5_testlong_penman)
# # print(statistics_train4, statistics_dev4, statistics_eval4, statistics_test4, statistics_train5, statistics_dev5,statistics_test5)
# #  of bars
# bar_width = 0.15
# ax.set_ylim([400,7700]) # numbers here are specific to this example
# ax2.set_ylim([0, 160])
# # Set the x-axis positions for each group
# x_group1 = np.arange(len(keys_group1))
# x_group2 = x_group1 + bar_width
# x_group3 = x_group2 + bar_width
# x_group4 = x_group3 + bar_width
#
# # Create histograms
# b1 = ax.bar(x_group1, values_group1, width=bar_width, label='Train')
# b2 = ax.bar(x_group2, values_group2, width=bar_width, label='Dev')
# b3 = ax.bar(x_group3, values_group3, width=bar_width, label='Test')
# b4 = ax.bar(x_group4, values_group4, width=bar_width, label='TestLong')
#
# ax.bar_label(b1)
# ax.bar_label(b2)
# ax.bar_label(b3)
# ax.bar_label(b4)
#
# b5 = ax2.bar(x_group1, values_group1, width=bar_width, label='Train')
# b6 =ax2.bar(x_group2, values_group2, width=bar_width, label='Dev')
# b7 =ax2.bar(x_group3, values_group3, width=bar_width, label='Test')
# b8 = ax2.bar(x_group4, values_group4, width=bar_width, label='TestLong')
#
# ax2.bar_label(b5)
# ax2.bar_label(b6)
# ax2.bar_label(b7)
# ax2.bar_label(b8)
# # Set x-axis labels
# plt.xlabel('Number of Boxes')
# plt.xticks(x_group1 + bar_width * 1.5, keys_group1)
#
# # Set y-axis label
# plt.ylabel('Count')
#
# # Set plot title
# plt.title('Total Number of Multibox Graphs Across Four Data Splits in PMB5 Dataset')
#
# # Add legend
# plt.legend()
#
# # Show the plot
# plt.tight_layout()
# plt.show()

