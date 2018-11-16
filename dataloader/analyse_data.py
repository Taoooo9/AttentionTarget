import re


def analyse(file):
    sen_len = []
    sen_dict = {}
    pos_num = 0
    neu_num = 0
    neg_num = 0
    pos = 0
    neu = 0
    neg = 0
    count = 0

    with open(file, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if re.findall('-positive', line):
                pos += 1
            elif re.search('-neutral', line):
                neu += 1
            elif re.search('-negative', line):
                neg += 1
            count += 1
            if len(line) == 0:
                if pos:
                    pos_num += 1
                elif neu:
                    neu_num += 1
                elif neg:
                    neg_num += 1
                sen_len.append(count - 1)
                count = 0
                pos = 0
                neu = 0
                neg = 0
    sen_len = sorted(sen_len, reverse=True)
    count1 = 1
    first_num = sen_len[0]
    for i, num in enumerate(sen_len[1:], 1):
        if first_num == num:
            count1 += 1
        else:
            sen_dict[first_num] = count1
            first_num = num
            count1 = 1
    print("样本一共有：{0}句".format(len(sen_len)))
    print("积极的句子有：{0}句".format(pos_num))
    print("中立的句子有：{0}句".format(neu_num))
    print("消极的句子有：{0}句".format(neg_num))
    for k, v in sen_dict.items():
        print("长度为{0}的句子有：{1}句".format(k, v))
    print('\n\n')


