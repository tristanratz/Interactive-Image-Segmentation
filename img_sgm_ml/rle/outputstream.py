import math

import numpy as np

U32 = int(math.pow(2, 32))


class OutputStream:
    """
    Python implementation of the Output Stream by thi-ng
    https://github.com/thi-ng/umbrella/tree/develop/packages/bitstream/
    """
    buffer: np.array
    pos: int
    bit: int
    bitpos: int

    def __init__(self, size):
        self.buffer = np.empty(size, dtype=np.uint8)
        self.pos = 0
        self.bitpos = 0
        self.seek(0)

    def position(self):
        return self.bitpos

    def seek(self, pos: int):
        if pos >= len(self.buffer) << 3:
            print("seek pos out of bounds:", pos)

        self.pos = pos >> 3
        self.bit = 8 - (pos & 0x7)
        self.bitpos = pos

    def write(self, x: int, word_size: int = 1):
        if word_size > 32:
            hi = int(math.floor(x / U32))
            self.write(hi, word_size - 32)
            self.write(x - hi * U32, 32)
        elif word_size > 8:
            n: int = word_size & -8
            msb = word_size - n
            if msb > 0:
                self._write(x >> n, msb)
            n -= 8
            while n >= 0:
                self._write(x >> n, 8)
                n -= 8
        else:
            self._write(x, word_size)

    def write_bit(self, x):
        self.bit = self.bit - 1
        self.buffer[self.pos] = (self.buffer[self.pos] & ~(1 << self.bit)) | (x << self.bit)

        if self.bit == 0:
            self.ensure_size()
            self.bit = 8
            self.bitpos = self.bitpos + 1

    def write_words(self, data, word_size):
        for v in data:
            self.write(v, word_size)

    def _write(self, x: int, word_size: int):
        x &= (1 << word_size) - 1
        buf = self.buffer
        pos = self.pos
        bit = self.bit
        b = bit - word_size
        m = ~((1 << bit) - 1) if bit < 8 else 0
        if b >= 0:
            m |= (1 << b) - 1
            buf[pos] = (buf[pos] & m) | ((x << b) & ~m)
            if b == 0:
                self.ensure_size()
                self.bit = 8
            else:
                self.bit = b
        else:
            self.bit = bit = 8 + b
            buf[pos] = (buf[pos] & m) | ((x >> -b) & ~m)
            self.ensure_size()
            self.buffer[self.pos] = (self.buffer[self.pos] & ((1 << bit) - 1)) | ((x << bit) & 0xff)
        self.bitpos += word_size

    def bytes(self):
        return self.buffer[0:self.pos + (1 if self.bit & 7 else 0)]

    def ensure_size(self):
        self.pos = self.pos + 1
        if self.pos == len(self.buffer):
            b = np.empty(len(self.buffer) << 1, dtype=np.uint8)
            b[:len(self.buffer)] = self.buffer
            self.buffer = b
