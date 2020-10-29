from img_sgm_ml.rle.outputstream import OutputStream


def encode(
        src: [int],
        num: [int],
        word_size=8,
        rle_sizes=None
) -> []:
    """
    Python implementation of thi-ng's rle-pack
    https://github.com/thi-ng/umbrella/blob/develop/packages/rle-pack/

    Args:
        src: The array to encode
        num: number of input words
        word_size: in bits, range 1 - 32
        rle_sizes: Four numbers [int,int,int,int]

    Returns: Encoded array

    """

    if rle_sizes is None:
        rle_sizes = [3, 4, 8, 16]

    if word_size < 1 or word_size > 32:
        print("Error: word size (1-32 bits only)")
        return

    out = OutputStream(int(round((num * word_size) / 8)) + 4 + 2 + 1)
    out.write(num, 32)
    out.write(word_size - 1, 5)
    for x in rle_sizes:
        if x < 1 or x > 16:
            print("Error: RLE repeat size (1-16 bits only)")
            return
        out.write(x - 1, 4)

    rle0: int = 1 << rle_sizes[0]
    rle1: int = 1 << rle_sizes[1]
    rle2: int = 1 << rle_sizes[2]
    rle3: int = 1 << rle_sizes[3]

    n1 = num - 1
    val = None
    tail = True
    i = 0
    chunk: [int] = []
    n = 0

    def write_rle(_n: int):
        t = 0 if _n < rle0 else 1 if _n < rle1 else 2 if _n < rle2 else 3
        out.write_bit(1)
        out.write(t, 2)
        out.write(_n, rle_sizes[t])
        out.write(val, word_size)

    def write_chunk(_chunk: [int]):
        m = len(_chunk) - 1
        t = 0 if m < rle0 else 1 if m < rle1 else 2 if m < rle2 else 3
        out.write_bit(0)
        out.write(t, 2)
        out.write(m, rle_sizes[t])
        out.write_words(_chunk, word_size)

    for x in src:
        if val is None:
            val = x
        elif x != val:
            if n > 0:
                write_rle(n)
                n = 0
            else:
                chunk.append(val)
                if len(chunk) == rle3:
                    write_chunk(chunk)
                    chunk = []
            val = x
        else:
            if len(chunk):
                write_chunk(chunk)
                chunk = []
            n = n + 1
            if n == rle3:
                n = n - 1
                write_rle(n)
                n = 0
                tail = i < n1
        if i == n1:
            break
        i = i + 1
    if len(chunk):
        chunk.append(val)
        write_chunk(chunk)
        chunk = []
    elif tail:
        write_rle(n)
        n = 0
    return out.bytes()
