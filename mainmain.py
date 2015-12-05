__author__ = 'constancedeperrois'

from main import myfunctionsuperkey as supersuper
import time

def myfunction(t, g, s, l):
    supersuper(t, g, s, l)


if __name__ == "__main__":
    test = raw_input("Loading complete. Start? y/n ")

    if test == "y":

        part = raw_input("Part of experiment: 1 or 2 ? If 2, make sure part 1 has properly been computed. ")

        start_time = time.time()

        if part == '1':
            myfunction(0, 3, 0, 0)
            myfunction(0, 4, 0, 0)
            myfunction(0, 0, 3, 0)
            #myfunction(0, 2, 1, 0)
            myfunction(0, 0, 0, 1)
            #myfunction(0, 0, 0, 2)

        elif part == '2':
            myfunction(0, 0, 0, 2)

        else:
            print "error : user input"

    else:
        print "nothing"
    print("--- %s seconds ---" % (time.time() - start_time))
