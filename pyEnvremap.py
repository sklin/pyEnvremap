#!/usr/bin/env python

#from cv2 import *
from numpy import *
from math import *

PI2 = 1.5707963
PI = 3.1415927
TAU = 6.2831853

### utils

def normalize(v):
    k = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    v[0] *= k
    v[1] *= k
    v[2] *= k
    #v *= k

def clamp(f, a, z):
    if f < a:
        return a
    elif f > z:
        return z
    else:
        return f

def lerp(a, b, k):
    return a * (1.0 - k) + b * k

### patterns

rgss_pattern = [
    (0.125, 0.625),
    (0.375, 0.125),
    (0.625, 0.875),
    (0.875, 0.375)
]

### to_env

def cube_to_env(f, i, j, h, w):
    p = [
        [(  0,  0, -1), (  0, -1,  0), (  1,  0,  0)],
        [(  0,  0,  1), (  0, -1,  0), ( -1,  0,  0)],
        [(  1,  0,  0), (  0,  0,  1), (  0,  1,  0)],
        [(  1,  0,  0), (  0,  0, -1), (  0, -1,  0)],
        [(  1,  0,  0), (  0, -1,  0), (  0,  0,  1)],
        [( -1,  0,  0), (  0, -1,  0), (  0,  0, -1)]
    ]
    y = 2.0 * i / h - 1.0
    x = 2.0 * j / w - 1.0

    v = [
        p[f][0][0] * x + p[f][1][0] * y + p[f][2][0],
        p[f][0][1] * x + p[f][1][1] * y + p[f][2][1],
        p[f][0][2] * x + p[f][1][2] * y + p[f][2][2]
    ]

    # normalize
    normalize(v)

    return (True, v)

def rect_to_env(f, i, j, h, w):
    lat = PI2 - PI * i / h
    lon = TAU * j / w - PI

    v = [
        sin(lon) * cos(lat),
                   sin(lat),
        -cos(lon) * cos(lat)
    ]
    #v[0] =  sin(lon) * cos(lat)
    #v[1] =             sin(lat)
    #v[2] = -cos(lon) * cos(lat)

    return (True, v)

### to_img

def rect_to_img(h, w, v):
    f = 0
    i = h * (              acos(v[1]) / PI )
    j = w * (0.5 + atan2(v[0], -v[2]) / TAU)
    return (True, (f, i, j))

def cube_to_img(h, w, v):
    X = abs(v[0])
    Y = abs(v[1])
    Z = abs(v[2])

    if (v[0] > 0 and X >= Y and X >= Z):
        f = 0
        x = -v[2] / X
        y = -v[1] / X
    elif (v[0] < 0 and X >= Y and X >= Z):
        f = 1
        x =  v[2] / X
        y = -v[1] / X
    elif (v[1] > 0 and Y >= X and Y >= Z):
        f = 2
        x =  v[0] / Y
        y =  v[2] / Y
    elif (v[1] < 0 and Y >= X and Y >= Z):
        f = 3
        x =  v[0] / Y
        y = -v[2] / Y
    elif (v[2] > 0 and Z >= X and Z >= Y):
        f = 4
        x =  v[0] / Z
        y = -v[1] / Z
    elif (v[2] < 0 and Z >= X and Z >= Y):
        f = 5
        x = -v[0] / Z
        y = -v[1] / Z
    else:
        return (False, ())

    i = 1.0 + (h - 2) * (y + 1.0) / 2.0
    j = 1.0 + (w - 2) * (x + 1.0) / 2.0

    return (True,(f, i, j))

### filter

def filter_linear(img, i, j, p):
    # img: numpy.array([height][width][3]), the image of the srcImg (input)
    # (i, j): the pixel is (i, j) of the image (input)
    # p: the pixel data (p[0:3]) for dstImg (result)
    ii = clamp(i-0.5, 0.0, size(img, 0) - 1.0)
    jj = clamp(j-0.5, 0.0, size(img, 1) - 1.0)

    i0, i1 = int(floor(ii)), int(ceil(ii))
    j0, j1 = int(floor(jj)), int(ceil(jj))

    di = ii - i0
    dj = jj - j0

    for k in xrange(len(p)):
        p[k] += lerp(
                    lerp(img[i0][j0][k], img[i0][j1][k], dj),
                    lerp(img[i1][j0][k], img[i1][j1][k], dj),
                    di
                )
    return

### main functions

def supersample(src, dst, pat, rot, fil, to_img, to_env, f, i, j):
    # f: the f th image in the dst[]
    c = 0
    p = dst[f][i][j]

    # For each sample of the supersampling pattern...
    for k in xrange(len(pat)):
        ii = pat[k][0] + i
        jj = pat[k][1] + j

        # Project and unproject giving the source location. Sample there.
        ret, v = to_env(f, i, j, size(dst[f], 0), size(dst[f], 1))
        if ret is False: return False

        ret, (F, I, J) = to_img(size(src[0], 0), size(src[0], 1), v)
        if ret is False: return False

        fil(src[F], I, J, p)
        c += 1

    # Normalize the sample
    p /= c # TODO: need to clamp(0, 255) ?

    return True

def process(src, dst, pat, rot, fil, to_img, to_env):
    # src: [img, img, img, ...], length depends on input type
    # dst: [img, img, img, ...], length depends on output type
    if len(src) <= 0 or len(dst) <= 0:
        return False

    for i in xrange(size(dst[0],0)):     # height of the image
        for j in xrange(size(dst[0],1)): # width of the image
            print "Handling dst[0][%d][%d]" % (i, j)
            for f in xrange(len(dst)):   # how many images in src
                supersample(src, dst, pat, rot, fil, to_img, to_env, f, i, j)
    return True

def sphere2cube(src, size):
    # input: 1 image
    # size: output image size
    # output: 6 images
    dst = [
        zeros((size, size, 3)),
        zeros((size, size, 3)),
        zeros((size, size, 3)),
        zeros((size, size, 3)),
        zeros((size, size, 3)),
        zeros((size, size, 3))
    ]

    rot = [0, 0, 0]

    process(src, dst, rgss_pattern, rot, filter_linear, rect_to_img, cube_to_env)
    return dst

def cube2sphere(src, size):
    # input: 6 images
    # output: 1 image
    dst = [
        zeros((size, size*2, 3)),
    ]

    rot = [0, 0, 0]

    process(src, dst, rgss_pattern, rot, filter_linear, cube_to_img, rect_to_env)
    return dst

def main():
    a = zeros((1024, 1024, 3))
    src = []
    src.append(a.copy())
    src.append(a.copy())
    src.append(a.copy())
    src.append(a.copy())
    src.append(a.copy())
    src.append(a.copy())
    cube2sphere(src, 64)


# size ? n ?
