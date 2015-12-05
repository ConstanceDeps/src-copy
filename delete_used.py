__author__ = 'constancedeperrois'

import os

# Delete all my data for graphs / just the babies / just the siblings

def main():

    nodes = raw_input("Enter number of nodes of the graph to delete : ")
    typeofdata = raw_input("Type of data to delete. r = results from simul, t = trackdis files, g = graphs files, p = pdfgraphs ")

    if typeofdata == 'r':
        directory = 'simres/'
        file1 = 'results' + nodes
        for filename in os.listdir(directory):
            if filename.startswith(file1):
                deletefile(filename, directory)

    elif typeofdata == 't':
        directory = 'trackres/'
        file2 = nodes + 'trackdis'
        for filename in os.listdir(directory):
            if filename.startswith(file2):
                deletefile(filename, directory)

    elif typeofdata == 'g':
        directory = 'graphdata/'
        file1 = 'edges' + nodes + '_'
        file2 = 'nodes' + nodes + '_'
        for filename in os.listdir(directory):
            if filename.startswith(file1) or filename.startswith(file2):
                deletefile(filename, directory)

    elif typeofdata == 'p':
        file1 = 'graph' + nodes + '_'
        for filename in os.listdir('graph_output' + '/'):
            if filename.startswith(file1):
                deletefile(filename, 'graph_output')

def deletefile(filename, directory):
    try:
        os.remove(directory + '/' + filename)
    except OSError:  ## if failed, report it back to the user ##
        print "Error with this file", filename


if __name__ == "__main__":
    main()
