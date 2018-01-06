#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps
import zipfile


def loadCSVintoSparse (filePath, header = False, separator="::"):

    values, rows, cols = [], [], []

    fileHandle = open(filePath, "r")
    numCells = 0

    if header:
        fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if (numCells % 1000000 == 0):
            print("Processed {} cells".format(numCells))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")

            if not line[2] == "0" and not line[2] == "NaN":
                rows.append(int(line[0]))
                cols.append(int(line[1]))
                values.append(float(line[2]))

    return  sps.csr_matrix((values, (rows, cols)), dtype=np.float32)



def saveSparseIntoCSV (filePath, sparse_matrix, separator=","):

    sparse_matrix = sparse_matrix.tocoo()

    fileHandle = open(filePath, "w")

    for index in range(len(sparse_matrix.data)):
        fileHandle.write("{row}{separator}{col}{separator}{value}\n".format(
            row = sparse_matrix.row[index], col = sparse_matrix.col[index], value = sparse_matrix.data[index],
            separator = separator))



class Movielens10MReader(object):

    def __init__(self, splitTrainTest = False, trainPercentage = 0.8, loadPredefinedTrainTest = True):

        super(Movielens10MReader, self).__init__()

        print("Movielens10MReader: loading data...")

        dataSubfolder = "./ml-10M100K/"

        URM_path = dataSubfolder + "ratings.dat"


        if not loadPredefinedTrainTest:
            self.URM_all = loadCSVintoSparse(URM_path, separator="::")

        else:

            try:
                self.URM_train = loadCSVintoSparse(dataSubfolder + "URM_train.csv", separator=",")
                self.URM_test = loadCSVintoSparse(dataSubfolder + "URM_test.csv", separator=",")

                return

            except FileNotFoundError:
                # Rebuild split
                print("Movielens10MReader: URM_train or URM_test not found. Building new ones")

                splitTrainTest = True
                self.URM_all = loadCSVintoSparse(URM_path)



        if splitTrainTest:

            self.URM_all = self.URM_all.tocoo()

            numInteractions= len(self.URM_all.data)

            mask = np.random.choice([True, False], numInteractions, p=[trainPercentage, 1 - trainPercentage])


            self.URM_train = sps.coo_matrix((self.URM_all.data[mask], (self.URM_all.row[mask], self.URM_all.col[mask])))
            self.URM_train = self.URM_train.tocsr()

            mask = np.logical_not(mask)

            self.URM_test = sps.coo_matrix((self.URM_all.data[mask], (self.URM_all.row[mask], self.URM_all.col[mask])))
            self.URM_test = self.URM_test.tocsr()

            del self.URM_all

            print("Movielens10MReader: saving URM_train and URM_test")
            saveSparseIntoCSV(dataSubfolder + "URM_train.csv", self.URM_train)
            saveSparseIntoCSV(dataSubfolder + "URM_test.csv", self.URM_test)

        print("Movielens10MReader: loading complete")




    def get_URM_train(self):
        return self.URM_train

    def get_URM_test(self):
        return self.URM_test
