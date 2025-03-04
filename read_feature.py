import re, sys, os, platform
import itertools
from collections import Counter

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
father_path = os.path.abspath(
    os.path.dirname(pPath) + os.path.sep + ".") + r'\pubscripts' if platform.system() == 'Windows' else os.path.abspath(
    os.path.dirname(pPath) + os.path.sep + ".") + r'/pubscripts'
sys.path.append(father_path)


def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer



def Kmer(fastas, k=2, type="DNA", upto=False, normalize=True, **kw):
    encoding = []
    header = ['#', 'label']
    NA = 'ACGT'
    if type in ("DNA", 'RNA'):
        NA = 'ACGT'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            sequence, label = re.sub('-', '', i[0]), i[1]
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            code = [label, i[2]]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            sequence, label = re.sub('-', '', i[0]), i[1]
            kmers = kmerArray(sequence, k)
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = [label, i[2]]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    return encoding

def read_nucleotide_sequences(file):
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip().replace(" ","") for line in lines]
    lines = lines[1:]

    fasta_sequences = []
    for fasta in lines:
        array = fasta.split(',')
        sequence = re.sub('[^ACGTU-]', '-', ''.join(array[0]).upper())
        sequence = re.sub('U', 'T', sequence)
        fasta_sequences.append([sequence, int(array[1]), array[0]]) # [[seq,label]]
    return fasta_sequences


def encode_feature(filePath:str):
    res = read_nucleotide_sequences(filePath)
    res3 = Kmer(res, k=3, upto=False, normalize=False)
    res4 = Kmer(res, k=4, upto=False, normalize=False)
    res5 = Kmer(res, k=5, upto=False, normalize=False)
    result = []
    for i in range(len(res3)):
        if i == 0:
            continue
        result.append(res3[i] + res4[i][2:] + res5[i][2:])
    return result


if __name__ == "__main__":
    res = encode_feature("./mouse_RNA.csv")
    print(res[5])
