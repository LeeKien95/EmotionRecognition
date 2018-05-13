import Models
import math
import face_recognition
import base64

def getSubjectPredict(imageString):
  subject_images = Models.getSubjectImages()
  imgdata = base64.b64decode(imageString)
  imgname = 'data/received_img.jpg'
  with open(imgname, 'wb') as f:
    f.write(imgdata)
  f.close()

  known_faces = []
  for si in subject_images:
    known_faces.append(si[0])
  unknown_face = face_recognition.load_image_file('data/received_img.jpg')
  unknown_face_encoding = face_recognition.face_encodings(unknown_face)[0]

  compare_results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
  result = []
  for i in range(0, len(compare_results)):
    if(compare_results[i] == True):
      if(subject_images[i][1] not in result):
        result.append(subject_images[i][1])
  print(result)
  return result

