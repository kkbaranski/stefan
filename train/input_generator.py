#!/usr/bin/env python
import struct
import os
import argparse
import random
import sys

from array import array
from PIL import Image

class Mnist:
    def __init__(self, source=None, path_img=os.path.join(os.path.dirname(os.path.realpath(__file__)),'train-images-idx3-ubyte'), 
                        path_lbl=os.path.join(os.path.dirname(os.path.realpath(__file__)),'train-labels-idx1-ubyte')):
        self.source = source
        if source is None:
            self.images, self.labels = load(path_img, path_lbl)

    def generate(self, test_mode=True, count=None, nr=None, expected=None):
        if self.source is None and nr is not None and nr not in range(len(self.images)): raise AttributeError
        if count is None:
            count = 1 if self.source is not None or nr is not None else len(self.images)
        else:
            count = count if self.source is not None or nr is not None else min(count, len(self.images))

        for i in range(count):
            x = nr if nr is not None else i
            if test_mode or (self.source is not None and expected is None): 
                print('1')
            else: 
                print('2 %d' % (expected if expected is not None else self.labels[x]))
            image = self.images[x] if self.source is None else self.source
            for p in range(28):
                row = image[28*p:28*p+28]
                row.append(0)
                print(' '.join('%.4f' % (y / 255.0) for y in row))
            print(' '.join('%.4f' % 0 for y in range(29)))
        print(0)

    def show(self, nr):
        image = self.source if self.source is not None else self.images[nr]
        im = Image.new("L", (28, 28))
        im.putdata(image)
        im = im.resize((112,112))
        im.show()


def load(path_img=os.path.join(os.path.dirname(os.path.realpath(__file__)),'train-images-idx3-ubyte'), 
        path_lbl=os.path.join(os.path.dirname(os.path.realpath(__file__)),'train-labels-idx1-ubyte')):
    with open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049: raise ValueError
        labels = array("B", file.read())

    with open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051: raise ValueError
        image_data = array("B", file.read())
    images = []
    for i in range(size): images.append([0] * rows * cols)
    for i in range(size): images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]
    if len(images) != len(labels): raise AttributeError
    return images, labels

def convert_from_file(path):
    source = list(Image.open(path).convert("RGB").convert("L").resize((28,28)).getdata())
    source = [255 - x for x in source]
    return source

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action='store_true', help="test mode (do not include expected value)")
    parser.add_argument('-s', action='store_true', help="show image (with option -i or -p)")
    parser.add_argument('-n', type=int, help="number of samples")
    parser.add_argument('-i', type=int, help="sample index")
    parser.add_argument('-p', type=str, metavar='PATH', help="generate from file")
    parser.add_argument('-e', type=int, help="expected value")
    return parser.parse_args()



def main():
    args = parse_args()

    mnist = Mnist(source=convert_from_file(args.p) if args.p is not None else None)
    if args.s:
        mnist.show(args.i)
    else:
        mnist.generate(test_mode=args.t, count=args.n, nr=args.i, expected=args.e)
   

if __name__ == '__main__':
    main()
