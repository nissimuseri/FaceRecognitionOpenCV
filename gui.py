from Tkinter import *
from PIL import Image, ImageTk


root = Tk()

root.title('FaceReco')
root.iconbitmap('icon.ico')
root.geometry("500x500")

image = Image.open('Background.jpg').resize((700,700) , Image.ANTIALIAS)
photo_image = ImageTk.PhotoImage(image)
label = Label(root, image = photo_image)
label.place(x=0, y=0, relwidth=1, relheight=1)

Title = Label(root, text="FaceReco")
Title.config(font=("David", 60), foreground="red", background="white")
Title.pack()

nameOfFirstStudents = Label(root, text= "Nissim Museri")
nameOfFirstStudents.place(x=0, y=480)
nameOfFirstStudents.config(font=("David", 12))

nameOfSecondStudents = Label(root, text= "Karin Krupetsky")
nameOfSecondStudents.place(x=388, y=480)
nameOfSecondStudents.config(font=("David", 12))

nameOfSecondStudents = Label(root, text="Ver: 2.0")
nameOfSecondStudents.place(x=220, y=480)
nameOfSecondStudents.config(font=("David", 8))


def openURL(url):
    import webbrowser
    webbrowser.open_new(url)


playButton = Button(root, text="Play Simulation", cursor="hand2")
playButton.place(x=135, y=350)
playButton.config(font=("David", 24))
playButton.bind("<Button-1>", lambda e: openURL("https://youtu.be/gMQWggxtoUM"))


def play():
    import Algo


playButton = Button(root, text="Try Now", command=play)
playButton.place(x=145, y=150)
playButton.config(font=("David", 35))


root.mainloop()
