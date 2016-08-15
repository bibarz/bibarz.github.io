import collections
import csv
import cv2
import scipy.weave
import numpy as np
import time
import os
import random



if __name__ == "__main__":
    nrec = 100000
    with open("/media/psf/Home/linux-home/Borja/Cursos/kaggle/bimbo/train.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile)
        headings = reader.next()
        important_headings = ['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID',
                              'Producto_ID', 'Venta_uni_hoy', 'Dev_uni_proxima', 'Demanda_uni_equil']
        important_idx = [headings.index(s) for s in important_headings]
        m = np.empty((nrec, len(important_idx)), dtype=np.int)
        for i in range(nrec):
            row = reader.next()
            m[i] = [int(row[x]) for x in important_idx]
    print important_headings
    print np.vstack((np.amax(m, axis=0), np.amin(m, axis=0)))
    import pdb;pdb.set_trace()
