

# this is a helper file to find accuracy from command-line data running it needs command-line data
# to be in specific format


with open(r"results") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line


res_2 = []
res_3 = []
res_4 = []
res_5 = []
res_z = []


def maxindex(a, b, c, d, e):
    # finds maximum value
    li = (a, b, c, d, e)
    res = li.index(max(li))
    return res


res_check_list = []
# read the data
for i in range(60):
    res_2.append(float(content[i].split(' ')[2]))
    res_3.append(float(content[i + 60].split(' ')[2]))
    res_4.append(float(content[i + 120].split(' ')[2]))
    res_5.append(float(content[i + 180].split(' ')[2]))
    res_z.append(float(content[i + 240].split(' ')[2]))
    res_check_list.append(i // 12)

mistakes = 0
# classify and check accuracy
for i in range(len(res_2)):
    if res_check_list[i] != maxindex(res_2[i], res_3[i], res_4[i], res_5[i], res_z[i]):
        mistakes += 1

print('accuracy is', (1 - (mistakes / 60)) * 100)
