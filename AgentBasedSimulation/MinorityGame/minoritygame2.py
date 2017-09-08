#! /usr/bin/env python3

def  Player( mode ):
    """  Player( mode ) zwraca funkcję gracza w trybie mode, gdzie
         mode = 0, 1, 2, 3
         Dla mode=0,1 gracz powtarza zawsze 0 lub 1, dla mode=2,3
         gracza powtarza lub neguje poprzedni wynik

         Funkcja gracza wywoływana jest z poprzednim (historycznym)
         wynikiem gry (jako 0 lub 1), ma zwrócić decyzję gracza, jako 0 lub 1.
    """
    def  pfun0( prev ):
        return 0

    def  pfun1( prev ):
        return 1

    def  pfun2( prev ):
        return  prev

    def  pfun3( prev ):
        return  1 - prev

    return (pfun0, pfun1, pfun2, pfun3)[mode]

def  game( a, b, c, len ):
    """  Wykonanie gry o dlugości len
         a, b, c - gracze
    """
    prev = 0
    score = [0, 0, 0]

    for i in range(0, len):
        bid1 = a( prev )
        bid2 = b( prev )
        bid3 = c( prev )

        count = 0
        if bid1: count += 1
        if bid2: count += 1
        if bid3: count += 1

        # Co wygrywa - prev = głos mniejszości
        prev = 1 if count<2 else 0

        if (bid1 == prev):
            score[0] += 1

        if (bid2 == prev):
            score[1] += 1

        if (bid3 == prev):
            score[2] += 1

    return score

if __name__=="__main__":
    total = [0, 0, 0, 0]

    for i in 0,1,2,3:
        for j in 0,1,2,3:
            for k in 0,1,2,3:
                score = game( Player(i), Player(j), Player(k), 41 )
                print( i, score[0] )
                print( j, score[1] )
                print( k, score[2] )

                total[ i ] += score[0]
                total[ j ] += score[1]
                total[ k ] += score[2]


    print( total )
