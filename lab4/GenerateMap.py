import random


def genSearchField(size):
    searchField = []
    for _ in range(0, size):
        a = [random.randint(0, 1) for _ in range(0, size)]
        searchField.append(a)

    searchField[5][5] = 10 #WYNIK

    return searchField

