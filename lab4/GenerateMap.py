import random


def genSearchField(size, targetsNumber):
    searchField = []
    for _ in range(0, size):
        a = [0 for _ in range(0, size)]
        searchField.append(a)

    searchField = genTargets(searchField, size, targetsNumber)

    return searchField


def genTargets(searchField, size, targetsNumber):
    sFC = searchField
    targetValue = 10

    for i in range(0, targetsNumber):
        pos1 = random.randint(0, size-1)
        pos2 = random.randint(0, size-1)

        sFC[pos1][pos2] = targetValue
        targetValue += 10

    return sFC
