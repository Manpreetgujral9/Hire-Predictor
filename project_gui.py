from tkinter import *

import numpy as np
import pandas as pd
from PIL import ImageTk, Image
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
excel_file = 'HiredDataSet.xlsx'
df = pd.read_excel(excel_file)
print(df)
print("Dataset Length:: ", len(df))
print("Dataset Shape:: ", df.shape)
features = list(df.columns[:5])
print(features)

def Logistic_reg():
    Y = df["Hire?"]
    X = df[features]
    Y=np.ravel(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)
    bb =int(b.get())
    cc = int(c.get())
    dd = int(d.get())
    ee = int(e.get())
    ff = int(f.get())
    res = np.array([[bb, cc, dd, ee, ff]])
    result=(model2.predict(res))
    print(result)
    if (result == 1):
        log_reg.set('Hired')
    else:
        log_reg.set('Not Hired')

def random_forest():
    Y = df["Hire?"]
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X, Y)
    bb =int(b.get())
    cc = int(c.get())
    dd = int(d.get())
    ee = int(e.get())
    ff = int(f.get())
    res = np.array([[bb, cc, dd, ee, ff]])
    result=(clf.predict(res))
    print(result)
    if (result == 1):
        random_for.set('Hired')
    else:
        random_for.set('Not Hired')

def Decision_tree():
    Y = df["Hire?"]
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    bb = b.get()
    cc = c.get()
    dd = d.get()
    ee = e.get()
    ff = f.get()
    res = clf_gini.predict([[bb, cc, dd, ee, ff]])
    print(res)
    if (res == 1):
        dec_tree.set('Hired')
    else:
        dec_tree.set('Not Hired')


def svm():
    Y = df["Hire?"]
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf = SVC()
    clf.fit(X_train, y_train)
    SVC(kernel='linear', C=1, gamma=1)
    bb = b.get()
    cc = c.get()
    dd = d.get()
    ee = e.get()
    ff = f.get()
    res = np.array([[bb, cc, dd, ee, ff]])
    result = clf.predict(res)
    print(result)
    if (result == 1):
        svm_al.set('Hired')
    else:
        svm_al.set('Not Hired')



win = Tk()
win.geometry("1000x1000")
win.title("window application")
load = Image.open('imgg.jpg')
render = ImageTk.PhotoImage(load)
img = Label(win, image=render)
img.image = render
img.place(x=0, y=0)

frame = Frame(win, width=900, height=80, bd=8, relief='raised')
frame.pack(side='top')

Label(frame, text='Hire Predictor', font=('arial', 30, 'bold'), bg='white', fg='black', width=28).grid(row=0, column=0)

frame1 = Frame(win, width=1000, bg='white', bd=8, relief='raised', height=900, padx=10, pady=10)
frame1.pack(padx=20, pady=20)

lb = Label(frame1, text='Name:', font=('arial', 14, 'bold'), bg='white', fg='black')
lb.grid(row=0, column=0, padx=20, pady=15)
lb1 = Label(frame1, text='Percentage:', font=('arial', 14, 'bold'), bg='white', fg='black')
lb1.grid(row=1, column=0, pady=15)
lb1 = Label(frame1, text='Backlogs:', font=('arial', 14, 'bold'), bg='white', fg='black')
lb1.grid(row=2, column=0, pady=15)
lb1 = Label(frame1, text='Internship:', font=('arial', 14, 'bold'), bg='white', fg='black')
lb1.grid(row=3, column=0, pady=15)
lb1 = Label(frame1, text='FirstRound:', font=('arial', 14, 'bold'), bg='white', fg='black')
lb1.grid(row=4, column=0, pady=15)
lb1 = Label(frame1, text='Communication:', font=('arial', 14, 'bold'), bg='white', fg='black')
lb1.grid(row=5, column=0, pady=25)

lb1 = Label(frame1, text='Logistic Result', font=('arial', 14, 'bold'), bg='white', fg='black')
lb1.grid(row=6, column=0, pady=15)
lb1 = Label(frame1, text='Decision Tree Result', font=('arial', 14, 'bold'), bg='white', fg='black')
lb1.grid(row=6, column=1, padx=10, pady=10)
lb1 = Label(frame1, text='Random Forest Result', font=('arial', 14, 'bold'), bg='white', fg='black')
lb1.grid(row=6, column=2, padx=10, pady=10)
lb1 = Label(frame1, text='SVM Result', font=('arial', 14, 'bold'), bg='white', fg='black')
lb1.grid(row=6, column=3, padx=10, pady=10)

a = StringVar()
tb1 = Entry(frame1, textvariable=a, font=("arial", 12, 'bold'))
tb1.grid(row=0, column=1, ipadx=25, ipady=4)
b = StringVar()
tb2 = Entry(frame1, textvariable=b, font=("arial", 12, 'bold'))
tb2.grid(row=1, column=1, ipadx=25, ipady=4)
c = StringVar()
tb3 = Entry(frame1, textvariable=c, font=("arial", 12, 'bold'))
tb3.grid(row=2, column=1, ipadx=25, ipady=4)
d = StringVar()
tb4 = Entry(frame1, textvariable=d, font=("arial", 12, 'bold'))
tb4.grid(row=3, column=1, ipadx=25, ipady=4)
e = StringVar()
tb4 = Entry(frame1, textvariable=e, font=("arial", 12, 'bold'))
tb4.grid(row=4, column=1, ipadx=25, ipady=4)
f = StringVar()
tb4 = Entry(frame1, textvariable=f, font=("arial", 12, 'bold'))
tb4.grid(row=5, column=1, ipadx=25, ipady=4)

log_reg = StringVar()
tb4 = Entry(frame1,textvariable=log_reg, font=("arial", 12, 'bold'))
tb4.grid(row=7, column=0, padx=5, ipadx=15, ipady=15)
dec_tree = StringVar()
tb4 = Entry(frame1, textvariable=dec_tree, font=("arial", 12, 'bold'))
tb4.grid(row=7, column=1, padx=5, ipadx=15, ipady=15)
random_for=StringVar()
tb4 = Entry(frame1,textvariable=random_for, font=("arial", 12, 'bold'))
tb4.grid(row=7, column=2, padx=5, ipadx=15, ipady=15)
svm_al = StringVar()
tb4 = Entry(frame1, textvariable=svm_al, font=("arial", 12, 'bold'))
tb4.grid(row=7, column=3, padx=5, ipadx=15, ipady=15)

final=StringVar()
tb4 = Entry(frame1, textvariable=final)
tb4.grid(row=9, columnspan=7, ipadx=5, ipady=8)

btn = Button(frame1, text="Logistic", font=("arial", 12, 'bold'), bg='gray63', fg='black', width=13,command=Logistic_reg)
btn.grid(row=1, column=2, pady=10, padx=10)
btn = Button(frame1, text="Decision Tree", font=("arial", 12, 'bold'), bg='gray63', fg='black', width=13,
             command=Decision_tree)
btn.grid(row=2, column=2, pady=10)
btn = Button(frame1, text="Random Forest", font=("arial", 12, 'bold'), bg='gray63', fg='black', width=13,command=random_forest)
btn.grid(row=3, column=2, pady=10)
btn = Button(frame1, text="SVM", font=("arial", 12, 'bold'), bg='gray63', fg='black', width=13, command=svm)
btn.grid(row=4, column=2, pady=10)

btn = Button(frame1, text="Final Result", font=("arial", 14, 'bold'), bg='gray63', fg='black', width=13)
btn.grid(row=8, columnspan=7, padx=5, pady=10)
win.mainloop()
