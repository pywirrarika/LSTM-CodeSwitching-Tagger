def get_gold(evalfile):
    eF = open(evalfile, 'r').read().split('\n')
    gold = {}
    for line in eF:
        line = line.split()
        words = []
        tags = []
        for token in line:
            (word, tag) = token.split('///')
            words.append(word)
            tags.append(tag)
        gold[" ".join(words)] = tags
    return gold

def evaluate(data,gold):
    correct = 0
    total = 0
    for d in data:
        key = d[0]
        predv = d[1]
        try:
            goldv = gold[key]
        except KeyError:
            continue
        for tag in zip(predv,goldv):
            if tag[0] == tag[1]:
                correct += 1
            total += 1

    accuracy = correct/total
    print('Accuracy:',accuracy)
    return accuracy



