import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def linear_theory(x_exp, y_exp):
    """
    :param x_exp: list -- experimental data for x-axis
    :param y_exp: list -- experimental data for y-axis
    :return: x_th, y_th -- lists with linear approximation of experimental data
    """
    k = np.polyfit(x_exp, y_exp, 1)
    m, M = min(x_exp), max(x_exp)
    x_th = np.arange(m - 0.05 * (M - m), M + 0.05 * (M - m), 0.0001 * (M - m))
    y_th = []
    for _ in range(0, len(x_th)):
        y_th.append(k[0] * x_th[_] + k[1])
    return x_th, y_th


def plot1(x_exp, y_exp, x_name, y_name, legend, x_err_formula_index=0, y_err_formula_index=0):
    """
    Function for drawing plot with one curve of points (x_exp, y_exp) with linear approximation and error-bars
    :param x_exp: list -- experimental data for x-axis
    :param y_exp: list -- experimental data for y-axis
    :param x_name: string -- name for x-axis
    :param y_name: string -- name for x-axis
    :param legend: string -- legend for plot
    :param x_err_formula_index: int -- index that represents number of standard formulas
                                       for calculating errors (see x_err_formula)
    :param y_err_formula_index: int -- index that represents number of standard formulas
                                       for calculating errors (see н_err_formula)
    :return: plot
    """
    x_th, y_th = linear_theory(x_exp, y_exp)
    xerr, yerr = error_function(x_exp, y_exp, x_err_formula_index, y_err_formula_index)
    plt.figure(figsize=(10, 5))
    plt.plot(x_th, y_th, label=legend)
    plt.errorbar(x_exp, y_exp, xerr, yerr, fmt=".k", label="Экспериментальные точки")

    plt.xlabel(x_name, fontsize=14)
    plt.ylabel(y_name, fontsize=14)
    plt.grid(True)
    plt.legend(loc='best', fontsize=12)
    plt.show()


def plot_3_on_1(x_exp1, y_exp1, x_name1, y_name1, legend1, x_err_formula_index1, y_err_formula_index1,
                x_exp2, y_exp2, x_name2, y_name2, legend2, x_err_formula_index2, y_err_formula_index2,
                x_exp3, y_exp3, x_name3, y_name3, legend3, x_err_formula_index3, y_err_formula_index3, title):
    """
    Function for drawing plot with three subplots each containing one curve of points (x_exp, y_exp)
    with linear approximation and error-bars
    :param x_exp1: list -- experimental data for x-axis on subplot №1
    :param y_exp1: list -- experimental data for y-axis on subplot №1
    :param x_name1: string -- name for x-axis on subplot №1
    :param y_name1: string -- name for y-axis on subplot №1
    :param legend1: string -- legend for subplot №1
    :param x_err_formula_index1: int -- index that represents number of standard formulas
                                        for calculating errors (see x_err_formula)
    :param y_err_formula_index1: int -- index that represents number of standard formulas
                                        for calculating errors (see y_err_formula)
    :param x_exp2: list -- experimental data for x-axis on subplot №2
    :param y_exp2: list -- experimental data for y-axis on subplot №2
    :param x_name2: string -- name for x-axis on subplot №2
    :param y_name2: string -- name for y-axis on subplot №2
    :param legend2: string -- legend for subplot №2
    :param x_err_formula_index2: int -- index that represents number of standard formulas
                                        for calculating errors (see x_err_formula)
    :param y_err_formula_index2: int -- index that represents number of standard formulas
                                        for calculating errors (see y_err_formula)
    :param x_exp3: list -- experimental data for x-axis on subplot №3
    :param y_exp3: list -- experimental data for y-axis on subplot №3
    :param x_name3: string -- name for x-axis on subplot №3
    :param y_name3: string -- name for y-axis on subplot №3
    :param legend3: string -- legend for subplot №3
    :param x_err_formula_index3: int -- index that represents number of standard formulas
                                        for calculating errors (see x_err_formula)
    :param y_err_formula_index3: int -- index that represents number of standard formulas
                                        for calculating errors (see y_err_formula)
    :param title: string -- title for hole plot
    :return: plot
    """
    x_th1, y_th1 = linear_theory(x_exp1, y_exp1)
    xerr1, yerr1 = error_function(x_exp1, y_exp1, x_err_formula_index1, y_err_formula_index1)

    x_th2, y_th2 = linear_theory(x_exp2, y_exp2)
    xerr2, yerr2 = error_function(x_exp2, y_exp2, x_err_formula_index2, y_err_formula_index2)

    x_th3, y_th3 = linear_theory(x_exp3, y_exp3)
    xerr3, yerr3 = error_function(x_exp3, y_exp3, x_err_formula_index3, y_err_formula_index3)

    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.suptitle(title)
    ax0 = axs[0]
    ax0.plot(x_th1, y_th1, label=legend1)
    ax0.errorbar(x_exp1, y_exp1, xerr1, yerr1, fmt=".k", label="Экспериментальные точки")
    ax0.set_xlabel(x_name1, fontsize=14)
    ax0.set_ylabel(y_name1, fontsize=14)
    ax0.grid(True)
    ax0.legend(loc='best', fontsize=12)

    ax1 = axs[1]
    ax1.plot(x_th2, y_th2, label=legend2)
    ax1.errorbar(x_exp2, y_exp2, xerr2, yerr2, fmt=".k", label="Экспериментальные точки")
    ax1.set_xlabel(x_name2, fontsize=14)
    ax1.set_ylabel(y_name2, fontsize=14)
    ax1.grid(True)
    ax1.legend(loc='best', fontsize=12)

    ax2 = axs[2]
    ax2.plot(x_th3, y_th3, label=legend3)
    ax2.errorbar(x_exp3, y_exp3, xerr3, yerr3, fmt=".k", label="Экспериментальные точки")
    ax2.set_xlabel(x_name3, fontsize=14)
    ax2.set_ylabel(y_name3, fontsize=14)
    ax2.grid(True)
    ax2.legend(loc='best', fontsize=12)
    plt.show()


def plot_3_in_1(x_exp1, y_exp1, legend1, x_err_formula_index1, y_err_formula_index1,
                x_exp2, y_exp2, legend2, x_err_formula_index2, y_err_formula_index2,
                x_exp3, y_exp3, legend3, x_err_formula_index3, y_err_formula_index3, x_name, y_name):
    """
    Function for drawing plot with three curves of points (x_exp, y_exp)
    with linear approximation and error-bars
    :param x_exp1: list -- experimental data for x-axis
    :param y_exp1: list -- experimental data for y-axis
    :param legend1: string -- legend for curve №1
    :param x_err_formula_index1: int -- index that represents number of standard formulas
                                        for calculating errors (see x_err_formula)
    :param y_err_formula_index1: int -- index that represents number of standard formulas
                                        for calculating errors (see y_err_formula)
    :param x_exp2: list -- experimental data for x-axis
    :param y_exp2: list -- experimental data for y-axis
    :param legend2: string -- legend for curve №2
    :param x_err_formula_index2: int -- index that represents number of standard formulas
                                        for calculating errors (see x_err_formula)
    :param y_err_formula_index2: int -- index that represents number of standard formulas
                                        for calculating errors (see y_err_formula)
    :param x_exp3: list -- experimental data for x-axis
    :param y_exp3: list -- experimental data for y-axis
    :param legend3: string -- legend for curve №3
    :param x_err_formula_index3: int -- index that represents number of standard formulas
                                        for calculating errors (see x_err_formula)
    :param y_err_formula_index3: int -- index that represents number of standard formulas
                                        for calculating errors (see y_err_formula)
    :param x_name:  string -- name for x-axis on plot
    :param y_name:  string -- name for y-axis on plot
    :return: plot
    """
    x_th1, y_th1 = linear_theory(x_exp1, y_exp1)
    xerr1, yerr1 = error_function(x_exp1, y_exp1, x_err_formula_index1, y_err_formula_index1)

    x_th2, y_th2 = linear_theory(x_exp2, y_exp2)
    xerr2, yerr2 = error_function(x_exp2, y_exp2, x_err_formula_index2, y_err_formula_index2)

    x_th3, y_th3 = linear_theory(x_exp3, y_exp3)
    xerr3, yerr3 = error_function(x_exp3, y_exp3, x_err_formula_index3, y_err_formula_index3)

    plt.figure(figsize=(10, 5))
    plt.plot(x_th1, y_th1, label=legend1)
    plt.errorbar(x_exp1, y_exp1, xerr1, yerr1, fmt=".k")

    plt.plot(x_th2, y_th2, label=legend2)
    plt.errorbar(x_exp2, y_exp2, xerr2, yerr2, fmt=".k")

    plt.plot(x_th3, y_th3, label=legend3)
    plt.errorbar(x_exp3, y_exp3, xerr3, yerr3, fmt=".k")

    plt.xlabel(x_name, fontsize=17)
    plt.ylabel(y_name, fontsize=17)
    plt.grid(True)
    plt.legend(loc='best', fontsize=15)
    plt.show()


def plot_4_in_1(x_exp1, y_exp1, legend1, x_err_formula_index1, y_err_formula_index1,
                x_exp2, y_exp2, legend2, x_err_formula_index2, y_err_formula_index2,
                x_exp3, y_exp3, legend3, x_err_formula_index3, y_err_formula_index3,
                x_exp4, y_exp4, legend4, x_err_formula_index4, y_err_formula_index4, x_name, y_name):
    """
    Function for drawing plot with fore curves of points (x_exp, y_exp)
    with linear approximation and error-bars
    :param x_exp1: list -- experimental data for x-axis
    :param y_exp1: list -- experimental data for y-axis
    :param legend1: string -- legend for curve №1
    :param x_err_formula_index1: int -- index that represents number of standard formulas
                                        for calculating errors (see x_err_formula)
    :param y_err_formula_index1: int -- index that represents number of standard formulas
                                        for calculating errors (see y_err_formula)
    :param x_exp2: list -- experimental data for x-axis
    :param y_exp2: list -- experimental data for y-axis
    :param legend2: string -- legend for curve №2
    :param x_err_formula_index2: int -- index that represents number of standard formulas
                                        for calculating errors (see x_err_formula)
    :param y_err_formula_index2: int -- index that represents number of standard formulas
                                        for calculating errors (see y_err_formula)
    :param x_exp3: list -- experimental data for x-axis
    :param y_exp3: list -- experimental data for y-axis
    :param legend3: string -- legend for curve №3
    :param x_err_formula_index3: int -- index that represents number of standard formulas
                                        for calculating errors (see x_err_formula)
    :param y_err_formula_index3: int -- index that represents number of standard formulas
                                        for calculating errors (see y_err_formula)
    :param x_exp4: list -- experimental data for x-axis
    :param y_exp4: list -- experimental data for y-axis
    :param legend4: string -- legend for curve №4
    :param x_err_formula_index4: int -- index that represents number of standard formulas
                                        for calculating errors (see x_err_formula)
    :param y_err_formula_index4: int -- index that represents number of standard formulas
                                        for calculating errors (see y_err_formula)
    :param x_name: string -- name for x-axis on plot
    :param y_name: string -- name for y-axis on plot
    :return: plot
    """
    x_th1, y_th1 = linear_theory(x_exp1, y_exp1)
    xerr1, yerr1 = error_function(x_exp1, y_exp1, x_err_formula_index1, y_err_formula_index1)

    x_th2, y_th2 = linear_theory(x_exp2, y_exp2)
    xerr2, yerr2 = error_function(x_exp2, y_exp2, x_err_formula_index2, y_err_formula_index2)

    x_th3, y_th3 = linear_theory(x_exp3, y_exp3)
    xerr3, yerr3 = error_function(x_exp3, y_exp3, x_err_formula_index3, y_err_formula_index3)

    x_th4, y_th4 = linear_theory(x_exp4, y_exp4)
    xerr4, yerr4 = error_function(x_exp4, y_exp4, x_err_formula_index4, y_err_formula_index4)

    plt.figure(figsize=(10, 5))
    plt.plot(x_th1, y_th1, label=legend1)
    plt.errorbar(x_exp1, y_exp1, xerr1, yerr1, fmt=".k")

    plt.plot(x_th2, y_th2, label=legend2)
    plt.errorbar(x_exp2, y_exp2, xerr2, yerr2, fmt=".k")

    plt.plot(x_th3, y_th3, label=legend3)
    plt.errorbar(x_exp3, y_exp3, xerr3, yerr3, fmt=".k")

    plt.plot(x_th4, y_th4, label=legend4)
    plt.errorbar(x_exp4, y_exp4, xerr4, yerr4, fmt=".k")

    plt.xlabel(x_name, fontsize=17)
    plt.ylabel(y_name, fontsize=17)
    plt.grid(True)
    plt.legend(loc='best', fontsize=15)
    plt.show()


def x_err_formula(x_err_formula_index, x, y):
    """
    Function for calculating error for x-coordinate
    :param x_err_formula_index: int -- index that represents particular function error_x (x, y, ather parametrs)
                                       realized in x_err_formula function
    :param x: float -- x-coordinate from experimental data for calculating errors
    :param y: float -- y-coordinate from experimental data for calculating errors
    :return: float -- error of x-coordinate
    """
    if x_err_formula_index == 0:
        varepsilon = 0
        return abs(x) * varepsilon


def y_err_formula(y_err_formula_index, x, y):
    """
    Function for calculating error for y-coordinate
    :param y_err_formula_index: int -- index that represents particular function error_y (x, y, ather parametrs)
                                       realized in y_err_formula function
    :param x: float -- x-coordinate from experimental data for calculating errors
    :param y: float -- y-coordinate from experimental data for calculating errors
    :return: float -- error of y-coordinate
    """
    if y_err_formula_index == 0:
        varepsilon = 0
        return abs(y) * varepsilon


def error_function(x_exp, y_exp, x_err_formula_index, y_err_formula_index):
    """
    General function for calculating errors tha use particular formulas from x_err_formula and y_err_formula
    :param x_exp: list -- experimental data (x-coordinate)
    :param y_exp: list -- experimental data (y-coordinate)
    :param x_err_formula_index: int -- index that represents number of standard formulas
                                        for calculating errors (see x_err_formula)
    :param y_err_formula_index: int -- index that represents number of standard formulas
                                        for calculating errors (see y_err_formula)
    :return: x_err, y_err -- lists with errors for x- and y-coordinates
    """
    x_err = []
    y_err = []
    for _ in range(0, len(x_exp)):
        x_err.append(x_err_formula(x_err_formula_index, x_exp[_], y_exp[_]))
        y_err.append(y_err_formula(y_err_formula_index, x_exp[_], y_exp[_]))
    return x_err, y_err


def error_of_exp(x_exp, y_exp, flag=0):
    """
    Calculating errors of linear coefficients in experiment
    :param x_exp: list -- experimental data (x-coordinate)
    :param y_exp: list -- experimental data (y-coordinate)
    :param flag: int -- if flag == 0 function will print errors of linear coefficients in experiment
    :return: er_k, er_b -- float -- errors of linear coefficients in experiment
    """
    coefficient = np.polyfit(x_exp, y_exp, 1)
    k, b = coefficient[0], coefficient[1]
    av_x = 0
    for _ in range(len(x_exp)):
        av_x += x_exp[_]
    av_x = av_x / len(x_exp)

    av_y = 0
    for _ in range(len(y_exp)):
        av_y += y_exp[_]
    av_y = av_y / len(y_exp)

    D_x = 0
    for _ in range(len(x_exp)):
        D_x += (x_exp[_] - av_x)**2
    D_x = D_x / len(x_exp)

    D_y = 0
    for _ in range(len(y_exp)):
        D_y += (y_exp[_] - av_y) ** 2
    D_y = D_y / len(y_exp)

    av_x2 = 0
    for _ in range(len(x_exp)):
        av_x2 += x_exp[_]**2
    av_x2 = av_x2 / len(x_exp)

    er_k = np.sqrt(1/(len(x_exp)-2)*((D_y/D_x)-k**2))
    er_b = er_k * np.sqrt(av_x2)
    if flag == 0:
        print('Coefficions calculeted in linear approximation:')
        print("k = ", k, "+-", er_k)
        print("b = ", b, "+-", er_b)
    if flag == 1:
        return er_k, er_b


# def data_reader(name, contacts, B):
#     data = pd.read_csv(name)
#     delta_ = data[(data.contacts == contacts) & (abs(data['-B, T']) == B)]['U, mV'].tolist()
#     delta = []
#     for i in range(0, len(delta_) - 1):
#         if i % 2 == 0:
#             delta.append(delta_[i] - delta_[i + 1])
#     I_ = data[(data.contacts == contacts) & (abs(data['-B, T']) == B)]['I, mA'].tolist()
#     I = []
#     for i in range(0, len(I_) - 1):
#         if i % 2 == 0:
#             I.append(I_[i])
#     return I, delta


# def calculation(x_exp, y_exp, B, er_B):
#     k = np.polyfit(x_exp, y_exp, 1)
#     er_k, er_b = error_of_exp(x_exp, y_exp, 10)
#     h = 50e-9  # толщина образца
#     epsilon = ((er_k / k[0])**2 + (er_B / B)**2)**0.5
#     delta = epsilon * abs(k[0] * h / B)
#     print('R_H =', k[0] * h / B, '+-', delta)
#     print('epsilon =', epsilon)


data = pd.read_csv("data.csv")

I_ = data['I,mA'].tolist()
H = []
I = []
Nt0 = 1750
Nt1 = 300
Ns0 = 825
Ns1 = 435
d = 0.07
ls = 0.8
dIs = 1.4567
dxs = 0.096
D = 0.1
dt = 0.01
mu0 = 4e-7 * np.pi
dB = []
deltaB =[]
B0 = 0
for _ in range(len(I_)):
    I.append(float(I_[_]) * 0.001)
    H.append(-I[_] * Nt0 / (np.pi * D))

dx_ = data['dX,cm'].tolist()
deltax_=data["Deltax"].tolist()
dx = []
deltax = []
for _ in range(len(dx_)):
    dx.append(float(dx_[_]) * 0.01)
    deltax.append(float(deltax_[_]) * 0.01)
for i in range(len(dx)):
    dB.append(mu0 * Ns0 * (Ns1 / Nt1) * (d/dt)**2 * (dIs / ls) * (dx[i] / dxs))
    deltaB.append(mu0 * Ns0 * (Ns1 / Nt1) * (d/dt)**2 * (dIs / ls) * (deltax[i] / dxs))
B_ = []
deltaB_ = [0]
for i in range(len(dB)):
    if i == 0:
        B_.append(B0)
    else:
        B_.append(B_[i - 1] + dB[i])
        deltaB_.append(deltaB_[i - 1] + deltaB[i])
        #deltaB_.append(deltaB[i])
# for i in range(len(deltaB_)):
#     deltaB_[i]=abs(deltaB_[i]-deltaB_[round(len(dB)/4)])
B1 = max(B_)/2
B = []
for i in range(len(dB)):
    if i == 0:
        B.append(-B1)
    else:
        B.append(B[i - 1] + dB[i])
print(H)
print(B)

data_in = pd.read_csv("data_in.csv")
Ii_ = data_in['I,mA'].tolist()
Ii = []
Hi = []
dBi = []
deltaBi =[]
for _ in range(len(Ii_)):
    Ii.append(float(Ii_[_]) * 0.001)
    Hi.append(Ii[_] * Nt0 / (np.pi * D))

dxi_ = data_in['dX'].tolist()
deltaXi = data_in["DeltadX"].tolist()
dxi = []
dHi = []
for _ in range(len(dxi_)):
    dxi.append(float(dxi_[_]) * 0.01)
    deltaXi[_]*=0.01
    if _ == 0:
        dHi.append(Hi[_])
    else:
        dHi.append(Hi[_] - H[_ - 1])
        deltaXi[_]=deltaXi[_-1]+deltaXi[_]
for i in range(len(dxi)):
    dBi.append(mu0 * Ns0 * (Ns1 / Nt1) * (d/dt)**2 * (dIs / ls) * (dxi[i] / dxs))
    deltaBi.append(mu0 * Ns0 * (Ns1 / Nt1) * (d/dt)**2 * (dIs / ls) * (deltaXi[i] / dxs))
mu = []

for i in range(1, len(dBi)):
    mu.append(dBi[i] / dHi[i] * (1/mu0))


print("mu")
print(mu)
print(max(mu))
print(dBi[6])
print(deltaBi[6]-deltaBi[5])
Bi = []
for i in range(len(dBi)):
    if i == 0:
        Bi.append(B0)
    else:
        Bi.append(Bi[i - 1] + dBi[i])

Bmax = 1.766
Hmax = 8.14 * 1000
Ms = Bmax/mu0 - Hmax
print(Bmax/mu0 - Hmax)
mb = 927e-26
print(Ms / mb)

plt.figure(figsize=(10, 5))
plt.errorbar(H, B, marker="o",  color="black", label='Петля гистерезиса', yerr=deltaB_)
plt.errorbar(Hi, Bi, marker="o",  color="red", label='Кривая намагничивания', yerr=deltaBi)
plt.xlabel('H', fontsize=14)
plt.ylabel('B', fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.show()


