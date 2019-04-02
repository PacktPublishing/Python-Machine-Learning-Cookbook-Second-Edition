import face_recognition

Image1 = face_recognition.load_image_file("giuseppe.jpg")
Image2 = face_recognition.load_image_file("tiziana.jpg")
UnknownImage = face_recognition.load_image_file("tiziana2.jpg")

try:
    Image1Encoding = face_recognition.face_encodings(Image1)[0]
    Image2Encoding = face_recognition.face_encodings(Image2)[0]
    UnknownImageEncoding = face_recognition.face_encodings(UnknownImage)[0]
except IndexError:
    print("Any face was located. Check the image files..")
    quit()

known_faces = [
    Image1Encoding,
    Image2Encoding
]

results = face_recognition.compare_faces(known_faces, UnknownImageEncoding)

print("Is the unknown face a picture of Giuseppe? {}".format(results[0]))
print("Is the unknown face a picture of Tiziana? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))