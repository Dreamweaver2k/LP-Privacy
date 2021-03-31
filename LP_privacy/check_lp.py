import numpy as np

def merge(lp, lp2, aligned):
    match = 2
    mismatch = -1
    indel = -2
    scores = np.zeros((len(lp) + 1, len(lp2) + 1))
    backtrack = np.chararray((len(lp) + 1, len(lp2) + 1))
    alignment1 = ''
    alignment2 = ''

    scores[0][0] = 0
    for i in range(1, len(lp) + 1):
        scores[i][0] = scores[i-1][0] + indel
        backtrack[i][0] = 'u'
    for j in range(1, len(lp2) + 1):
        scores[0][j] = scores[0][j-1] + indel
        backtrack[0][j] = 'l'


    for i in range(1, len(lp) + 1):
        for j in range(1, len(lp2) + 1):
            if lp[i-1] is lp2[j-1]: diagonal = scores[i-1][j-1] + match
            else: diagonal = scores[i-1][j-1] + mismatch
            left = scores[i-1][j] + indel
            up = scores[i][j-1] + indel
            scores[i][j] = max(diagonal, left, up)
            if scores[i][j] == diagonal: backtrack[i][j] = 'd'
            elif scores[i][j] == left: backtrack[i][j] = 'u'
            else: backtrack[i][j] = 'l'


    n = len(lp)
    m = len(lp2)
    new_aligned = ['']*len(aligned)
    while (n != 0 or m != 0):
        if backtrack[n][m] == b'd':
            alignment1 += lp[n-1]
            alignment2 += lp2[m-1]
            for j in range(len(aligned)):
                new_aligned[j] += aligned[j][n-1]
            n = n-1
            m = m-1
        elif backtrack[n][m] == b'l':
            alignment1 += '-'
            alignment2 += lp2[m-1]
            for j in range(len(aligned)):
                new_aligned[j] += '-'
            m = m-1
        else:
            alignment2 += '-'
            alignment1 += lp[n-1]
            for j in range(len(aligned)):
                new_aligned[j] += aligned[j][n-1]
            n = n-1

    for j in range(len(new_aligned)): new_aligned[j] = new_aligned[j][::-1]
    alignment2 = alignment2[::-1]
    alignment1 = alignment1[::-1]
    new_aligned.append(alignment2)
    return alignment1, new_aligned

def check(lps, plate_number, coordinates):
    if plate_number not in lps:
        newplate = True
        plate = plate_number
        for p in lps:
            if lps[p].distanceTo(coordinates[0], coordinates[1]) < 200:
                newplate = False
                plate = p
    else:
        newplate = False
        plate = plate_number

    return newplate, plate

    