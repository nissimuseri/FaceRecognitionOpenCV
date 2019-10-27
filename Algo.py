
import cv2
import numpy as np
import face_recognition as fr

#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# Get a reference to web camera #0 (the default one)
videoCapture = cv2.VideoCapture(0)

# Load pictures and save 128-dimension face encoding for each face.
BarakObamaImage = fr.load_image_file("Images/Obama.jpg") #load an image to numpy array(using PIL).
BarakObamaFaceEncoding = fr.face_encodings(BarakObamaImage)[0] #returns 128-dimension face encoding for each face.


# Do the same thing to the another faces.
DonaldTrumpImage = fr.load_image_file("Images/Trump.jpg")
DonaldTrumpFaceEncoding = fr.face_encodings(DonaldTrumpImage)[0]

PutinTrumpImage = fr.load_image_file("Images/Putin.jpg")
PutinTrumpFaceEncoding = fr.face_encodings(PutinTrumpImage)[0]

JasonStathemImage = fr.load_image_file("Images/Stathem.jpg")
JasonStathemFaceEncoding = fr.face_encodings(JasonStathemImage)[0]

DwayneJohnsonImage = fr.load_image_file("Images/Johnson.jpg")
DwayneJohnsonFaceEncoding = fr.face_encodings(DwayneJohnsonImage)[0]

NissimMuseriImage = fr.load_image_file("Images/Museri.jpg")
NissimMuseriFaceEncoding = fr.face_encodings(NissimMuseriImage)[0]

KarinKrupetskyImage = fr.load_image_file("Images/Krupetsky.jpg")
KarinKrupetskyFaceEncoding = fr.face_encodings(KarinKrupetskyImage)[0]

DmitryPatashovatashovImage = fr.load_image_file("Images/Patashov.jpg")
DmitryPatashovFaceEncoding = fr.face_encodings(DmitryPatashovatashovImage)[0]

YuvalHamevulbalImage = fr.load_image_file("Images/Mevulbal.jpg")
YuvalHamevulbalFaceEncoding = fr.face_encodings(YuvalHamevulbalImage)[0]

EyalGolanImage = fr.load_image_file("Images/Golan.jpg")
EyalGolanFaceEncoding = fr.face_encodings(EyalGolanImage)[0]

MoshePeretzImage = fr.load_image_file("Images/Peretz.jpg")
MoshePeretzFaceEncoding = fr.face_encodings(MoshePeretzImage)[0]

OmerAdamImage = fr.load_image_file("Images/Adam.jpg")
OmerAdamFaceEncoding = fr.face_encodings(OmerAdamImage)[0]

BibiNetanyahuImage = fr.load_image_file("Images/Bibi.jpg")
BibiNetanyahuFaceEncoding = fr.face_encodings(BibiNetanyahuImage)[0]

BeniGantzImage = fr.load_image_file("Images/Gantz.jpg")
BeniGantzFaceEncoding = fr.face_encodings(BeniGantzImage)[0]

GalGadotImage = fr.load_image_file("Images/Gadot.jpg")
GalGadotFaceEncoding = fr.face_encodings(GalGadotImage)[0]

SaraNetanyahuImage = fr.load_image_file("Images/Sara.jpg")
SaraNetanyahuFaceEncoding = fr.face_encodings(SaraNetanyahuImage)[0]
# End of encoding known faces.

# Create arrays of known face encodings and their names.
knownFaceEncodings = [
    BarakObamaFaceEncoding,
    DonaldTrumpFaceEncoding,
    PutinTrumpFaceEncoding,
    JasonStathemFaceEncoding,
    DwayneJohnsonFaceEncoding,
    NissimMuseriFaceEncoding,
    KarinKrupetskyFaceEncoding,
    DmitryPatashovFaceEncoding,
    YuvalHamevulbalFaceEncoding,
    EyalGolanFaceEncoding,
    MoshePeretzFaceEncoding,
    OmerAdamFaceEncoding,
    BibiNetanyahuFaceEncoding,
    BeniGantzFaceEncoding,
    GalGadotFaceEncoding,
    SaraNetanyahuFaceEncoding

]
# Match name for each face.
knownFaceNames = [
    "Barak Obama",
    "Donald Trump",
    "Vladimir Putin",
    "Jason Stathem",
    "Dwayne Johnson",
    "Nissim Museri",
    "Karin Krupetsky",
    "Dmitry Patashov",
    "Yuval Hamevulbal",
    "Eyal Golan",
    "Moshe Peretz",
    "Omer Adam",
    "Bibi Netanyahu",
    "Beni Gantz",
    "Gal Gadot",
    "Sara Netanyahu"
]

# Initialize some variables.
face_locations = []
faceEncodings = []
faceNames = []
processThisFrame = True

while True:
    # Grab a single frame of video.
    ret, frame = videoCapture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing,
    # after all we will return to the original size.
    smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses).
    rgb_smallFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time.
    if processThisFrame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = fr.face_locations(rgb_smallFrame)
        face_encodings = fr.face_encodings(rgb_smallFrame, face_locations)

        faceNames = []
        for faceEncoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = fr.compare_faces(knownFaceEncodings, faceEncoding)
            name = "Unknown"

            # # If a match was found in knownFaceEncodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = knownFaceNames[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            faceDistances = fr.face_distance(knownFaceEncodings, faceEncoding)
            bestMatchIndex = np.argmin(faceDistances)
            if matches[bestMatchIndex]:
                name = knownFaceNames[bestMatchIndex]

            faceNames.append(name)

    processThisFrame = not processThisFrame

    # Display the results.
    for (top, right, bottom, left), name in zip(face_locations, faceNames):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        if name == "Unknown":
            # Draw a red box around unknown faces.
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a red label with a name below unknown faces.
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        else:
            # Draw a green box around known faces
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw a green label with a name below known faces.
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        # Write the name of known faces or "Unknown" for unknown faces.
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image.
    cv2.imshow('Video', frame)

    # Press 'esc' to exit.
    if cv2.waitKey(1) == 27:
        break

# Release handle to the web camera.
videoCapture.release()
cv2.destroyAllWindows()
