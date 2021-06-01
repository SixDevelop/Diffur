import math
import matplotlib.pyplot as plt
import numpy as np
import time

c2 = 0.35  # init input data
A = 3
B = -3
C = 3


def f1(x, y2, y4):  # functions from system of differential equations
    return 2 * x * np.cbrt(y2) * y4


def f2(x, y3, y4):
    return 2 * B * x * np.exp(B / C * (y3 - A)) * y4


def f3(x, y4):
    return 2 * C * x * y4


def f4(x, y1):
    return -2 * x * np.log(y1)


y1_0 = 1  # Начальные условия
y2_0 = 1
y3_0 = A
y4_0 = 1
a = 0
b = 5
x_0 = a
tol = 0.00001

a21 = c2  # коэффициенты, полученные из условий порядка при s = 2
b2 = 1 / (2 * c2)
b1 = 1 - b2


def find_step(tol, p=2, x0=a, xk=b, y1_0=1, y2_0=1, y3_0=A, y4_0=1):        #вычисление оптимального начального шага
    f0 = np.array([f1(x0, y2_0, y4_0), f2(x0, y3_0, y4_0), f3(x0, y4_0), f4(x0, y1_0)])
    delta = pow(1 / max(abs(x0), abs(xk)), p + 1) + pow(np.linalg.norm(f0), p + 1)
    h1 = pow(tol / delta, 1 / (p + 1))
    K11 = h1 * f1(x0, y2_0, y4_0)  # вычисление первых производных
    K21 = h1 * f2(x0, y3_0, y4_0)
    K31 = h1 * f3(x0, y4_0)
    K41 = h1 * f4(x0, y1_0)
    K12 = h1 * f1(x0 + c2 * h1, y2_0 + a21 * K21, y4_0 + a21 * K41)  # вычисление вторых проихводных
    K22 = h1 * f2(x0 + c2 * h1, y3_0 + a21 * K31, y4_0 + a21 * K41)
    K32 = h1 * f3(x0 + c2 * h1, y4_0 + a21 * K41)
    K42 = h1 * f4(x0 + c2 * h1, y1_0 + a21 * K11)
    y1_0 = y1_0 + b1 * K11 + b2 * K12  # вычисление новых начальных условий
    y2_0 = y2_0 + b1 * K21 + b2 * K22
    y3_0 = y3_0 + b1 * K31 + b2 * K32
    y4_0 = y4_0 + b1 * K41 + b2 * K42
    x0 = x0 + h1
    u1 = np.array([y1_0, y2_0, y3_0, y4_0])     #вычисление шага по соответствующим формулам
    delta = pow(1 / max(abs(x0 + h1), abs(xk)), p + 1) + pow(np.linalg.norm(u1), p + 1)
    h1_new = pow(tol / delta, 1 / (p + 1))
    return min(h1, h1_new)


def Newton_Eiler(h, I, x=x_0, y1=y1_0, y2=y2_0, y3=y3_0, y4=y4_0, a = a, b = b):  # Метод Ньютона-Эйлера для с2 = 0.35
    X = [x]  # списки для построения графиков
    Y1 = [y1]
    Y2 = [y2]
    Y3 = [y3]
    Y4 = [y4]

    for i in range(int((b - a) / h) + 1):  # начинаем проделывать цикл по шагам

        K11 = h * f1(x, y2, y4)  # вычисление первых производных
        K21 = h * f2(x, y3, y4)
        K31 = h * f3(x, y4)
        K41 = h * f4(x, y1)

        K12 = h * f1(x + c2 * h, y2 + a21 * K21, y4 + a21 * K41)  # вычисление вторых производных
        K22 = h * f2(x + c2 * h, y3 + a21 * K31, y4 + a21 * K41)
        K32 = h * f3(x + c2 * h, y4 + a21 * K41)
        K42 = h * f4(x + c2 * h, y1 + a21 * K11)

        y1 = y1 + b1 * K11 + b2 * K12  # вычисление новых начальных условий
        y2 = y2 + b1 * K21 + b2 * K22
        y3 = y3 + b1 * K31 + b2 * K32
        y4 = y4 + b1 * K41 + b2 * K42
        x = x + h

        if (y1 <= 0):
            print("у1 не в области определения логарифма")
            break

        X.append(x)  # добавление вычисленных значений в массив для графика
        Y1.append(y1)
        Y2.append(y2)
        Y3.append(y3)
        Y4.append(y4)
        # print(x)

    f = np.array([y1, y2, y3, y4])  # вычисление нормы точной полной погрешности в конце отрезка
    y1_true = math.exp(math.sin(x ** 2))
    y2_true = math.exp(B * math.sin(x ** 2))
    y3_true = C * math.sin(x ** 2) + A
    y4_true = math.cos(x ** 2)
    f_true = np.array([y1_true, y2_true, y3_true, y4_true])
    norm_exact_error = np.linalg.norm(f - f_true)
    if I == 1:      #различные выводы функции для различных задач
        return -math.log(abs(norm_exact_error), 10)
    if I == 0:
        return X, Y1, Y2, Y3, Y4


def opponent26(h, I, x=x_0, y1=y1_0, y2=y2_0, y3=y3_0, y4=y4_0, a = a, b = b):  # Метод оппонент (26)
    X = [x]  # списки для построения графиков
    Y1 = [y1]
    Y2 = [y2]
    Y3 = [y3]
    Y4 = [y4]

    с2_op = 0.5  # коэффициенты, полученные из условий порядка для метода оппонент (26)
    a21_op = с2_op
    b2_op = 1 / (2 * с2_op)
    b1_op = 1 - b2_op

    for i in range(int((b - a) / h) + 1):  # начинаем проделывать цикл по шагам

        K11 = h * f1(x, y2, y4)  # вычисление первых производных
        K21 = h * f2(x, y3, y4)
        K31 = h * f3(x, y4)
        K41 = h * f4(x, y1)

        K12 = h * f1(x + h / 2, y2 + a21_op * K21, y4 + a21_op * K41)  # вычисление вторых проихводных
        K22 = h * f2(x + h / 2, y3 + a21_op * K31, y4 + a21_op * K41)
        K32 = h * f3(x + h / 2, y4 + a21_op * K41)
        K42 = h * f4(x + h / 2, y1 + a21_op * K11)

        y1 = y1 + b1_op * K11 + b2_op * K12  # вычисление новых начальных условий
        y2 = y2 + b1_op * K21 + b2_op * K22
        y3 = y3 + b1_op * K31 + b2_op * K32
        y4 = y4 + b1_op * K41 + b2_op * K42
        x = x + h

        if (y1 <= 0):
            print("у1 не в области определения логарифма")
            break

        X.append(x)  # добавление вычисленных значений в массив для графика
        Y1.append(y1)
        Y2.append(y2)
        Y3.append(y3)
        Y4.append(y4)
        # print(x)

    f = np.array([y1, y2, y3, y4])  # вычисление нормы точной полной погрешности в конце отрезка
    y1_true = math.exp(math.sin(x ** 2))
    y2_true = math.exp(B * math.sin(x ** 2))
    y3_true = C * math.sin(x ** 2) + A
    y4_true = math.cos(x ** 2)
    f_true = np.array([y1_true, y2_true, y3_true, y4_true])
    norm_exact_error = np.linalg.norm(f - f_true)
    if I == 1:      #различные выводы функции для различных задач
        return -math.log(abs(norm_exact_error), 10)
    if I == 0:
        return X, Y1, Y2, Y3, Y4


def Runge_sum_error(h, p=2):        #Оценка полной пограшности метода Ньютона_Эйлера
    X_h, Y1_h, Y2_h, Y3_h, Y4_h = Newton_Eiler(h, 0)
    X_h2, Y1_h2, Y2_h2, Y3_h2, Y4_h2 = Newton_Eiler(h / 2, 0)

    # print(Y1_h[-1], Y2_h[-1], Y3_h[-1], Y4_h[-1])
    # print(Y1_h2[-1], Y2_h2[-1], Y3_h2[-1], Y4_h2[-1])

    Rn1 = (Y1_h2[-1] - Y1_h[-1]) / (1 - 2 ** (-p))      #Вычисление компонент погрешности при шаге h
    Rn2 = (Y2_h2[-1] - Y2_h[-1]) / (1 - 2 ** (-p))
    Rn3 = (Y3_h2[-1] - Y3_h[-1]) / (1 - 2 ** (-p))
    Rn4 = (Y4_h2[-1] - Y4_h[-1]) / (1 - 2 ** (-p))

    R2n1 = (Y1_h2[-1] - Y1_h[-1]) / (2 ** (p) - 1)      #Вычисление компонент погрешности при шаге h/2
    R2n2 = (Y2_h2[-1] - Y2_h[-1]) / (2 ** (p) - 1)
    R2n3 = (Y3_h2[-1] - Y3_h[-1]) / (2 ** (p) - 1)
    R2n4 = (Y4_h2[-1] - Y4_h[-1]) / (2 ** (p) - 1)

    Rn = [Rn1, Rn2, Rn3, Rn4]       #Полная погрешность при шаге h
    R2n = [R2n1, R2n2, R2n3, R2n4]      #Полная погрешность при шаге h/2

    for i in range(0, 4):       #Выбираем, какой шаг далее будем использовать
        flag1 = 1
        flag2 = 1
        if (abs(Rn[i]) <= tol):
            print("Rn", i + 1, "удовлетворяет tol")
            print("Rn", i + 1, "=", Rn[i])
        else:
            print("Rn", i + 1, "не удовлетворяет tol")
            print("Rn", i + 1, "=", Rn[i])
            flag1 = 0
        if (abs(R2n[i]) <= tol):
            print("R2n", i + 1, "удовлетворяет tol")
            print("R2n", i + 1, "=", R2n[i])
        else:
            print("R2n", i + 1, "не удовлетворяет tol")
            print("R2n", i + 1, "=", R2n[i])
            flag2 = 0
    print(flag1, flag2)
    if (flag2 == 1):        #Предпочтительнее брать h/2
        print("Возвращаем значения h / 2, R2n, так как R2n удовлетворяет tol")
        return h / 2, R2n
    elif (flag1 == 1):
        print("Возвращаем значения h, Rn, так как Rn удовлетворяет tol (в отличие от Rn)")
        return h, Rn
    else:       #если никакое значение погрешности не удовлетворило tol, то пересчитываем значение шага заново
        Rn_norm = np.linalg.norm(Rn)
        h = h * pow(tol / abs(Rn_norm), 1 / p)
        print("Ни Rn, ни R2n не удовлетворяет tol, идет перерасчет шага h_tol")
        h, Rn = Runge_sum_error(h)
        return h, Rn

'''
Автоматический выбор шага интегрирования для метода Ньютона-Эйлера
'''
def Runge_autostep_NE(rtol, atol=1e-12, h=find_step(tol),  x=x_0, y1=y1_0, y2=y2_0, y3=y3_0, y4=y4_0, p=2):
    X_runge = [x]       #Массивы для построения графиков
    Y1_runge = [y1]
    Y2_runge = [y2]
    Y3_runge = [y3]
    Y4_runge = [y4]
    H = [h]
    R = [0]
    while b > X_runge[-1]:
        x = X_runge[-1]
        y1 = Y1_runge[-1]
        y2 = Y2_runge[-1]
        y3 = Y3_runge[-1]
        y4 = Y4_runge[-1]
        a_runge = X_runge[-1]
        b_runge = X_runge[-1]+h-1e-17
        """
        Делаем один шаг с шагом h и два с шагом h/2
        """
        X_h, Y1_h, Y2_h, Y3_h, Y4_h = Newton_Eiler(h, 0, x=x, y1=y1, y2=y2, y3=y3, y4=y4, a = a_runge, b = b_runge)
        X_h2, Y1_h2, Y2_h2, Y3_h2, Y4_h2 = Newton_Eiler(h/2, 0, x=x, y1=y1, y2=y2, y3=y3, y4=y4, a = a_runge, b = b_runge)

        # print("h", X_h)
        # print("2h", X_h2)
        # r1_1 = (Y1_h2[-1] - Y1_h[-1]) / (1 - 2 ** (-p))
        # r1_2 = (Y2_h2[-1] - Y2_h[-1]) / (1 - 2 ** (-p))
        # r1_3 = (Y3_h2[-1] - Y3_h[-1]) / (1 - 2 ** (-p))
        # r1_4 = (Y4_h2[-1] - Y4_h[-1]) / (1 - 2 ** (-p))

        r2_1 = (Y1_h2[-1] - Y1_h[-1]) / (2 ** (p) - 1)      #вычисляем компоненты локальной погрешности
        r2_2 = (Y2_h2[-1] - Y2_h[-1]) / (2 ** (p) - 1)
        r2_3 = (Y3_h2[-1] - Y3_h[-1]) / (2 ** (p) - 1)
        r2_4 = (Y4_h2[-1] - Y4_h[-1]) / (2 ** (p) - 1)

        # r1 = [r1_1, r1_2, r1_3, r1_4]
        r2 = [r2_1, r2_2, r2_3, r2_4]       #собираем компоненты вместе

        # y1_true = math.exp(math.sin(X_h[-1] ** 2))
        # y2_true = math.exp(B * math.sin(X_h[-1] ** 2))
        # y3_true = C * math.sin(X_h[-1] ** 2) + A
        # y4_true = math.cos(X_h[-1] ** 2)
        # Y_true = [y1_true, y2_true, y3_true, y4_true]
        # Y_true = [Y1_h2[-1], Y2_h2[-1], Y3_h2[-1], Y4_h2[-1]]
        Y_true = [Y1_h[-1], Y2_h[-1], Y3_h[-1], Y4_h[-1]]

        tol_min = np.min([rtol * np.linalg.norm(Y_true), atol])     #вычисляем минимальную допустимую погрешность

        # if(np.linalg.norm(r1) < tol_min):
        #     X_runge.append(X_h[-1])
        #     Y1_runge.append((Y1_h[-1]))
        #     Y2_runge.append((Y2_h[-1]))
        #     Y3_runge.append((Y3_h[-1]))
        #     Y4_runge.append((Y4_h[-1]))
        #     H.append(h)
        #     R.append(r1)
        #     print(X_runge[-1], "r1")
        if (np.linalg.norm(r2) < tol_min):      #проверяем, удовлетворяет ли наша погрешность допустимой
            X_runge.append(X_h2[-1])        #если да, то делаем шаг
            Y1_runge.append((Y1_h2[-1]))
            Y2_runge.append((Y2_h2[-1]))
            Y3_runge.append((Y3_h2[-1]))
            Y4_runge.append((Y4_h2[-1]))
            H.append(h / 2)
            R.append(r2)
            print(X_runge[-1], 'r2')
        else:       #если нет, то уменьшаем шаг
            # h = h * pow(tol_min / np.linalg.norm(r2), 1 / p)
            h = h * 0.9
            print("new h =", h)
    return X_runge, Y1_runge, Y2_runge, Y3_runge, Y4_runge, H, R


'''
Автоматический выбор шага интегрирования для метода оппонент
'''

def Runge_autostep_op(rtol, atol=1e-12, h=find_step(tol),  x=x_0, y1=y1_0, y2=y2_0, y3=y3_0, y4=y4_0, p=2):
    X_runge = [x]       #Массивы для построения графиков
    Y1_runge = [y1]
    Y2_runge = [y2]
    Y3_runge = [y3]
    Y4_runge = [y4]
    H = [h]
    R = [0]
    while b > X_runge[-1]:
        x = X_runge[-1]
        y1 = Y1_runge[-1]
        y2 = Y2_runge[-1]
        y3 = Y3_runge[-1]
        y4 = Y4_runge[-1]
        a_runge = X_runge[-1]
        b_runge = X_runge[-1]+h-1e-17
        """
        Делаем один шаг с шагом h и два с шагом h/2
        """
        X_h, Y1_h, Y2_h, Y3_h, Y4_h = opponent26(h, 0, x=x, y1=y1, y2=y2, y3=y3, y4=y4, a = a_runge, b = b_runge)
        X_h2, Y1_h2, Y2_h2, Y3_h2, Y4_h2 = opponent26(h/2, 0, x=x, y1=y1, y2=y2, y3=y3, y4=y4, a = a_runge, b = b_runge)

        # print("h", X_h)
        # print("2h", X_h2)
        # r1_1 = (Y1_h2[-1] - Y1_h[-1]) / (1 - 2 ** (-p))
        # r1_2 = (Y2_h2[-1] - Y2_h[-1]) / (1 - 2 ** (-p))
        # r1_3 = (Y3_h2[-1] - Y3_h[-1]) / (1 - 2 ** (-p))
        # r1_4 = (Y4_h2[-1] - Y4_h[-1]) / (1 - 2 ** (-p))

        r2_1 = (Y1_h2[-1] - Y1_h[-1]) / (2 ** (p) - 1)      #вычисляем компоненты локальной погрешности
        r2_2 = (Y2_h2[-1] - Y2_h[-1]) / (2 ** (p) - 1)
        r2_3 = (Y3_h2[-1] - Y3_h[-1]) / (2 ** (p) - 1)
        r2_4 = (Y4_h2[-1] - Y4_h[-1]) / (2 ** (p) - 1)

        # r1 = [r1_1, r1_2, r1_3, r1_4]
        r2 = [r2_1, r2_2, r2_3, r2_4]       #собираем компоненты вместе

        # y1_true = math.exp(math.sin(X_h[-1] ** 2))
        # y2_true = math.exp(B * math.sin(X_h[-1] ** 2))
        # y3_true = C * math.sin(X_h[-1] ** 2) + A
        # y4_true = math.cos(X_h[-1] ** 2)
        # Y_true = [y1_true, y2_true, y3_true, y4_true]
        # Y_true = [Y1_h2[-1], Y2_h2[-1], Y3_h2[-1], Y4_h2[-1]]
        Y_true = [Y1_h[-1], Y2_h[-1], Y3_h[-1], Y4_h[-1]]

        tol_min = np.min([rtol * np.linalg.norm(Y_true), atol])      #вычисляем минимальную допустимую погрешность

        # if(np.linalg.norm(r1) < tol_min):
        #     X_runge.append(X_h[-1])
        #     Y1_runge.append((Y1_h[-1]))
        #     Y2_runge.append((Y2_h[-1]))
        #     Y3_runge.append((Y3_h[-1]))
        #     Y4_runge.append((Y4_h[-1]))
        #     H.append(h)
        #     R.append(r1)
        #     print(X_runge[-1], "r1")
        if (np.linalg.norm(r2) < tol_min):      #проверяем, удовлетворяет ли наша погрешность допустимой
            X_runge.append(X_h2[-1])        #если да, то делаем шаг
            Y1_runge.append((Y1_h2[-1]))
            Y2_runge.append((Y2_h2[-1]))
            Y3_runge.append((Y3_h2[-1]))
            Y4_runge.append((Y4_h2[-1]))
            H.append(h / 2)
            R.append(r2)
            print(X_runge[-1], 'r2')
        else:       #если нет, то уменьшаем шаг
            # h = h * pow(tol_min / np.linalg.norm(r2), 1 / p)
            h = h * 0.9
            print("new h =", h)
    return X_runge, Y1_runge, Y2_runge, Y3_runge, Y4_runge, H, R



h_tol, Rn = Runge_sum_error(find_step(tol)) #вычисляем шаг через оценку полной погрешности для оценки точности метода
# h_tol = 1.8172594995827362e-05
# print("h_tol = ", h_tol)
X, Y1, Y2, Y3, Y4 = Newton_Eiler(h_tol, 0)      #выводим массивы для построения графиков
X_op, Y1_op, Y2_op, Y3_op, Y4_op = opponent26(h_tol, 0)

plt.title('Сравнение y1 истинного и приближенного')         #Построение графиков y1 истинного и приближенного
plt.xlabel('[a, b]')
plt.ylabel('y1')
plt.grid(True)
plt.plot(X, [math.exp(math.sin(x**2)) for x in X])
plt.plot(X_op, Y1_op)
plt.plot(X, Y1)
plt.legend(['Истинное решение','Метод оппонент','Метод Ньютона-Эйлера с с2 = 0.35'], loc=2)
plt.show()

plt.title('Сравнение y2 истинного и приближенного')         #Построение графиков y2 истинного и приближенного
plt.xlabel('[a, b]')
plt.ylabel('y2')
plt.grid(True)
plt.plot(X, [math.exp(B * math.sin(x**2)) for x in X])
plt.plot(X_op, Y2_op)
plt.plot(X, Y2)
plt.legend(['Истинное решение','Метод оппонент','Метод Ньютона-Эйлера с с2 = 0.35'], loc=2)
plt.show()

plt.title('Сравнение y3 истинного и приближенного')         #Построение графиков y3 истинного и приближенного
plt.xlabel('[a, b]')
plt.ylabel('y3')
plt.grid(True)
plt.plot(X, [C * math.sin(x ** 2) + A for x in X])
plt.plot(X_op, Y3_op)
plt.plot(X, Y3)
plt.legend(['Истинное решение','Метод оппонент','Метод Ньютона-Эйлера с с2 = 0.35'], loc=2)
plt.show()

plt.title('Сравнение y4 истинного и приближенного')         #Построение графиков y4 истинного и приближенного
plt.xlabel('[a, b]')
plt.ylabel('y4')
plt.grid(True)
plt.plot(X, [math.cos(x ** 2) for x in X])
plt.plot(X_op, Y4_op)
plt.plot(X, Y4)
plt.legend(['Истинное решение','Метод оппонент','Метод Ньютона-Эйлера с с2 = 0.35'], loc=2)
plt.show()


'''
Задание 1 - Построение графика зависимости нормы точной полной погрешности в конце отрезка от длины шага
в двойной логарифмической шкале
'''

ExactErrors_NE = []        #Построение графика зависимости нормы точной полной погрешности в конце отрезка
H = []                  #от длины шага в двойной логарифмической шкале
ExactErrors_op = []

for k in range(10, 4, -1):
    h = 1/(2**k)
    log_norm_NE = Newton_Eiler(h, 1)
    log_norm_op = opponent26(h, 1)
    H.append(h)
    ExactErrors_NE.append(log_norm_NE)
    ExactErrors_op.append(log_norm_op)

# print("h", H)
# print("log", ExactErrors_NE)

X_line = []     #Построение прямой для сравнения
Y_line = []
for x in range(0, 2):
    X_line.append(x)
    Y_line.append(2*x)


plt.title('Норма точной полной погрешности при h = 1/(2**k)')       #график lg(норма погрешности)
plt.xlabel('шаг h')
plt.ylabel('lg(норма погрешности)')
plt.grid(True)
plt.plot(H, ExactErrors_NE, 'y', H, ExactErrors_op, 'b', X_line, Y_line, 'g')
plt.legend(['Метод Ньютона-Эйлера','Метод оппонент','прямая'], loc=2)
plt.show()


# '''
# Задание 2 - построение графика зависимости нормы точной полной погрешности от независимой переменной при шаге h_opt
# '''
#
# F_norm = []     #Список для графика
# f_true = np.array([0, 0, 0, 0])
# f = np.array([0, 0, 0, 0])
# i = 0
# Y1_true = []        # Список значений точных решений
# Y2_true = []
# Y3_true = []
# Y4_true = []
# for x in X:
#     y1_true = math.exp(math.sin(x ** 2))
#     y2_true = math.exp(B * math.sin(x ** 2))
#     y3_true = C * math.sin(x ** 2) + A
#     y4_true = math.cos(x ** 2)
#     f_true = np.array([y1_true, y2_true, y3_true, y4_true])
#     f = np.array([Y1[i], Y2[i], Y3[i], Y4[i]])
#     i += 1
#     F_norm.append(np.linalg.norm(f - f_true))       #вычисление логарифма нормы погрешности
#     Y1_true.append(y1_true)
#     Y2_true.append(y2_true)
#     Y3_true.append(y3_true)
#     Y4_true.append(y4_true)
#
# plt.title('Зависимость нормы точной полной погрешности от х при шаге h_opt')         #Построение графика
# plt.xlabel('x')
# plt.ylabel('норма точной полной погрешности')
# plt.grid(True)
# plt.plot(X, F_norm)
# plt.show()
#
# '''
# Построение графика зависимости модулей погрешности для yi от х при шаге h_opt
# '''
# i = 0       #Строим их потому что графики решений лежат слишком близко друг к другу
# F_norm_y1 = []
# F_norm_y2 = []
# F_norm_y3 = []
# F_norm_y4 = []
# for x in X:         #Стараемся избегать ситуации, когда аргументом логарифма является 0
#     if (abs(Y1[i] - Y1_true[i] > 10**(-10))):
#         F_norm_y1.append(math.log(abs(Y1[i] - Y1_true[i]) ** (-1), 10) )
#     else:
#         F_norm_y1.append(abs(1e-17) ** (-1), 10))
#     if (abs(Y2[i] - Y2_true[i] > 10 ** (-10))):
#         F_norm_y2.append(math.log(abs(Y2[i] - Y2_true[i]) ** (-1), 10))
#     else:
#         F_norm_y2.append(abs(1e-17) ** (-1), 10))
#     if (abs(Y3[i] - Y3_true[i] > 10 ** (-10))):
#         F_norm_y3.append(math.log(abs(Y3[i] - Y3_true[i]) ** (-1), 10))
#     else:
#         F_norm_y3.append(abs(1e-17) ** (-1), 10))
#     if (abs(Y4[i] - Y4_true[i] > 10 ** (-10))):
#         F_norm_y4.append(math.log(abs(Y4[i] - Y4_true[i]) ** (-1), 10))
#     else:
#         F_norm_y4.append(abs(1e-17) ** (-1), 10))
#     i += 1
#
# plt.title('Погрешности для yi')         #Выводим сразу все графики точности
# plt.xlabel('[a, b]')
# plt.ylabel('lg|yi - yi_true|')
# plt.grid(True)
# plt.plot(X, F_norm_y1, X, F_norm_y2, X, F_norm_y3, X, F_norm_y4)
# plt.legend(['для y1','для y2','для y3', 'для y4'], loc=2)
# plt.show()
#


'''
Задание 3 - построение алгоритма с автоматическим выбором шага
'''
start_time = time.time()
X_runge, Y1_runge, Y2_runge, Y3_runge, Y4_runge, H, R = Runge_autostep_NE(rtol = 1e-8)
stop_time = time.time()
print("Время выполнения для авто выбора шага для метода Ньютона-Эйлера при rtol = 1e-6 -", stop_time - start_time)
# plt.title('Сравнение значений y1 c автоматическим выбором шага')         #график y1
# plt.xlabel('[a, b]')
# plt.ylabel('y1')
# plt.grid(True)
# plt.plot(X, [math.exp(math.sin(x**2)) for x in X])
# plt.plot(X_runge, Y1_runge)
# plt.legend(['Истинное решение', 'Алгоритм рунге для y1'], loc=2)
# plt.show()
#
# plt.title('Сравнение значений y2 c автоматическим выбором шага')         #график y2
# plt.xlabel('[a, b]')
# plt.ylabel('y2')
# plt.grid(True)
# plt.plot(X, [math.exp(B * math.sin(x**2)) for x in X])
# plt.plot(X_runge, Y2_runge)
# plt.legend(['Истинное решение', 'Алгоритм рунге для y2'], loc=2)
# plt.show()
#
# plt.title('Сравнение значений y3 c автоматическим выбором шага')         #график y3
# plt.xlabel('[a, b]')
# plt.ylabel('y3')
# plt.grid(True)
# plt.plot(X, [C * math.sin(x ** 2) + A for x in X])
# plt.plot(X_runge, Y3_runge)
# plt.legend(['Истинное решение', 'Алгоритм рунге для y3'], loc=2)
# plt.show()
#
#
# plt.title('Сравнение значений y4 c автоматическим выбором шага')        #графиков y4 и
# plt.xlabel('[a, b]')
# plt.ylabel('y4')
# plt.grid(True)
# plt.plot(X, [math.cos(x ** 2) for x in X])
# plt.plot(X_runge, Y4_runge)
# plt.plot(X, Y4)
# plt.legend(['Истинное решение', 'Алгоритм рунге для y4'], loc=2)
# plt.show()
#
i = 0
Y1_true = []            #Массивы с точным решением для построения графика погрешности
Y2_true = []
Y3_true = []
Y4_true = []
for x in X_runge:
    y1_true = math.exp(math.sin(x ** 2))
    y2_true = math.exp(B * math.sin(x ** 2))
    y3_true = C * math.sin(x ** 2) + A
    y4_true = math.cos(x ** 2)
    i += 1
    Y1_true.append(y1_true)
    Y2_true.append(y2_true)
    Y3_true.append(y3_true)
    Y4_true.append(y4_true)

i = 0
F_norm_y1 = []      #Массивы для построения графика логарифма нормы погрешности
F_norm_y2 = []
F_norm_y3 = []
F_norm_y4 = []
for x in X_runge:
    if (abs(Y1_runge[i] - Y1_true[i] > 10**(-10))):
        F_norm_y1.append(math.log(abs(Y1_runge[i] - Y1_true[i]) ** (-1), 10) )
    else:
        F_norm_y1.append(math.log(abs(1e-17) ** (-1), 10))
    if (abs(Y2_runge[i] - Y2_true[i] > 10 ** (-10))):
        F_norm_y2.append(math.log(abs(Y2_runge[i] - Y2_true[i]) ** (-1), 10))
    else:
        F_norm_y2.append(math.log(abs(1e-17) ** (-1), 10))
    if (abs(Y3_runge[i] - Y3_true[i] > 10 ** (-10))):
        F_norm_y3.append(math.log(abs(Y3_runge[i] - Y3_true[i]) ** (-1), 10))
    else:
        F_norm_y3.append(math.log(abs(1e-17) ** (-1), 10))
    if (abs(Y4_runge[i] - Y4_true[i] > 10 ** (-10))):
        F_norm_y4.append(math.log(abs(Y4_runge[i] - Y4_true[i]) ** (-1), 10))
    else:
        F_norm_y4.append(math.log(abs(1e-17) ** (-1), 10))
    i += 1

plt.title('Погрешности для yi при авто шаге')
plt.xlabel('[a, b]')
plt.ylabel('lg|yi - yi_true|')
plt.grid(True)
plt.plot(X_runge, F_norm_y1, X_runge, F_norm_y2, X_runge, F_norm_y3, X_runge, F_norm_y4)
plt.legend(['для y1','для y2','для y3', 'для y4'], loc=2)
plt.show()
#
# plt.title('Изменение шага в алгоритме с авто выбором шага')     #Посмотрим, как менялся шаг
# plt.xlabel('[a, b]')
# plt.ylabel('h')
# plt.grid(True)
# plt.plot(X_runge, H)
# plt.show()


