from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']

label_encoder.fit(input_classes)
print("Class mapping: ")
for i, item in enumerate(label_encoder.classes_):
    print(item, "-->", i)
    
labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print("Labels =", labels)
print("Encoded labels =", list(encoded_labels))

encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("Encoded labels =", encoded_labels)
print("Decoded labels =", list(decoded_labels))