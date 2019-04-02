from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file("ciaburro.jpg")

FaceLandmarksList = face_recognition.face_landmarks(image)

print("Number {} face(s) recognized in this image.".format(len(FaceLandmarksList)))

PilImage = Image.fromarray(image)
DrawPilImage = ImageDraw.Draw(PilImage)

for face_landmarks in FaceLandmarksList:

    for facial_feature in face_landmarks.keys():
        print("{} points: {}".format(facial_feature, face_landmarks[facial_feature]))

    for facial_feature in face_landmarks.keys():
        DrawPilImage.line(face_landmarks[facial_feature], width=5)

PilImage.show()