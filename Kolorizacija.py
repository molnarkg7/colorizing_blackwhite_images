import cv2
import numpy as np
import PySimpleGUI as sg
import os.path

# Definisanje korisnickog interfejsa
lista_slika = [
    [
        sg.Text("Izaberi sliku: "),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

pregled_slike = [
    [sg.Text("Izabrana slika: "),],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-SLIKA-")],
    [sg.Submit(key = '-SUBMIT-')]

]

layout = [
    [
        sg.Column(lista_slika),
        sg.VSeperator(),
        sg.Column(pregled_slike),
    ]
]

window = sg.Window("Kolorizacija", layout)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED or event == '-SUBMIT-':
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-SLIKA-"].update(filename=filename)

        except:
            pass

window.close()

prototxt= './model/colorization_deploy_v2.prototxt'
caffemodel = './model/colorization_release_v2.caffemodel'
pts = np.load('./model/pts_in_hull.npy')
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]


image = cv2.imread(filename)
scaled = image.astype("float32")/255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)


resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50


net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)

cv2.waitKey(0)
