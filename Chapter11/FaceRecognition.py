from PIL import Image
import face_recognition

image = face_recognition.load_image_file("family.jpg")

face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

print("Number {} face(s) recognized in this image.".format(len(face_locations)))

for face_location in face_locations:

    top, right, bottom, left = face_location
    print("Face location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()