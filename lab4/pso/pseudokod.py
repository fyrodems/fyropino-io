funkcja funkcja_celu(X):
    A = 10
    y = A * 2 + suma([(x^2 - A * cos(2 * pi * x)) dla x in X])
    zwróć y

funkcja f(x):
    liczba_cząstek = liczba_cząstek(x)
    j = [funkcja_celu(x[i]) dla i w zakresie liczba_cząstek]
    zwróć j

funkcja przeprowadź_eksperyment(liczba_cząstek, wymiary, opcje, iteracje, nazwa_eksperymentu):
    optymalizator = inicjalizuj_optymalizator(liczba_cząstek, wymiary, opcje)

    koszt, pozycje = optymalizuj_funkcję(optymalizator, iteracje)
    historia_kosztu = pobierz_historię_kosztu(optymalizator)

    narysuj_wykres_historii_kosztu(historia_kosztu)
    ustaw_tytuł_wykresu('Historia Kosztu - ' + nazwa_eksperymentu)
    ustaw_etykiety_osi('Iteracje', 'Koszt')
    zapisz_wykres('historia_kosztu_' + nazwa_eksperymentu + '.png')

    mesher = stwórz_mesher()
    animacja = narysuj_trajektorię_cząstek(optymalizator.pos_history, mesher, oznaczenie=(0, 0))
    zapisz_animację(animacja, 'wykres_' + nazwa_eksperymentu + '.gif')
    wyświetl_animację('wykres_' + nazwa_eksperymentu + '.gif')

funkcja inicjalizuj_optymalizator(liczba_cząstek, wymiary, opcje):
    zwróć GlobalBestPSO(liczba_cząstek=liczba_cząstek, wymiary=wymiary, opcje=opcje)

funkcja optymalizuj_funkcję(optymalizator, iteracje):
    zwróć optymalizator.optymalizuj(f, iteracje=iteracje)

funkcja pobierz_historię_kosztu(optymalizator):
    zwróć optymalizator.historia_kosztu

funkcja stwórz_mesher():
    zwróć Mesher(func=fx.sphere)

funkcja narysuj_trajektorię_cząstek(pos_history, mesher, oznaczenie):
    zwróć plot_contour(pos_history=pos_history, mesher=mesher, oznaczenie=oznaczenie)

funkcja zapisz_animację(animacja, nazwa_pliku):
    animacja.zapisz(nazwa_pliku, writer='imagemagick', fps=4)

funkcja wyświetl_animację(nazwa_pliku):
    wyświetl_obrazek(url=nazwa_pliku)

# Eksperyment 1: Wpływ liczby cząstek
opcje1 = {'c1': 1, 'c2': 2, 'w': 0.75}
przeprowadź_eksperyment(liczba_cząstek=500, wymiary=2, opcje=opcje1, iteracje=100, nazwa_eksperymentu='eksperyment1')

# Eksperyment 2: Wpływ parametrów algorytmu
opcje2 = {'c1': 1.5, 'c2': 1.5, 'w': 0.9}
przeprowadź_eksperyment(liczba_cząstek=800, wymiary=2, opcje=opcje2, iteracje=100, nazwa_eksperymentu='eksperyment2')

# Eksperyment 3: Wpływ inicjalizacji cząstek
opcje3 = {'c1': 0.5, 'c2': 1.8, 'w': 0.5}
przeprowadź_eksperyment(liczba_cząstek=1000, wymiary=2, opcje=opcje3, iteracje=100, nazwa_eksperymentu='eksperyment3')
