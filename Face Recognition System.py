import cv2
import face_recognition

# # Load images and learn how to recognize each person.
# known_images = [
#      "C:\\Users\\eranj\\OneDrive\\Desktop\\Python\\diksha.jpeg"
#  
known_images = [
    "C:\\Users\\eranj\\OneDrive\\Desktop\\Python\\Anuj.jpeg",
      "C:\\Users\\eranj\\OneDrive\\Desktop\\Python\\Arunima.jpeg",
     "C:\\Users\\eranj\\OneDrive\\Desktop\\Python\\bhakti.jpeg",
     "C:\\Users\\eranj\\OneDrive\\Desktop\\Python\\diksha.jpeg",
     "C:\\Users\\eranj\\OneDrive\\Desktop\\Python\\Geetika.jpeg"
]

# Load images using OpenCV with error checking
known_images_cv2 = [cv2.imread(img) for img in known_images]

known_images_rgb = []
for img in known_images_cv2:
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        known_images_rgb.append(img_rgb)
    else:
        print(f"Error loading image: {img}")

# Get face encodings

known_encodings=[]

for img in range(len(known_images_rgb)):
    known_encodings.append(face_recognition.face_encodings(known_images_rgb[img])[0])


# Create arrays of known face encodings and their corresponding names
known_face_encodings = known_encodings.copy()
known_face_names = ["Anuj", "Arunima Sethi", "Bhakti Goyal", "Diksha Khangarot", "Geetika Nag"]

face_cascade = cv2.CascadeClassifier("C:/Users/eranj/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)

while True: 
    ret, video_data = video_cap.read()
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) from the grayscale image
        roi_gray = col[y:y+h, x:x+w]
        
        # Check if the detected face matches any known face
        face_encodings = face_recognition.face_encodings(video_data, [(y, x+w, y+h, x)])
        name = "Unknown"
        if len(face_encodings) > 0:
            # Check for a match with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            if True in matches:
                name = known_face_names[matches.index(True)]

        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(video_data, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("video_live", video_data)
    
    if cv2.waitKey(10) == ord("a"):
        break

video_cap.release()
cv2.destroyAllWindows()