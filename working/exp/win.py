from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
usernameLabel=None
usernameInput=None
def showMessage():
    messagebox.showinfo("Hello", "Welcome")

def greetUser(event):
    username=usernameInput.get()
    usernameLabel['text'] = username

window = Tk()
window.geometry("500x500")

# label
label = Label(window, text='First label', bg='white', fg='blue', font=('Serif', 16))

# image
img = Image.open('/var/www/ai-py/working/images/vg_night.png')
img = img.resize((50, 50))
imgTk = ImageTk.PhotoImage(img)
image = Label(window, image=imgTk)

# entry
frame = Frame(window, bg='blue', height=100)
usernameLabel = Label(frame, text='Username', font=('Serif', 16))
usernameInput = Entry(frame)

# checkbox
frame2 = Frame(window, bg='blue', height=100)
check1 = Checkbutton(frame2, text='check1')
check2 = Checkbutton(frame2, text='check2')
check3 = Checkbutton(frame2, text='check3')
check4 = Checkbutton(frame2, text='check4')

# radio
frame3 = Frame(window, bg='blue', height=100)
var = StringVar(frame3, "1")
radio1 = Radiobutton(frame3, text="Radio1", variable=var, value="1")
radio2 = Radiobutton(frame3, text="Radio2", variable=var, value="2")

# button
button = Button(window, text="Let's go", command=showMessage)
button.bind("<Button-1>", greetUser)

# pack
label.pack()
image.pack()
frame.pack()
usernameLabel.pack(side=LEFT)
usernameInput.pack(side=RIGHT)
frame2.pack()
check1.pack()
check2.pack()
check3.pack()
check4.pack()
frame3.pack()
radio1.pack()
radio2.pack()
button.place(x=200, y=350)

window.mainloop()
