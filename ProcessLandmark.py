import Models
import math
from sklearn.preprocessing import normalize

def emotion_decode(argument):
    emotion = {
        1: "Anger",
        2: "Contemp",
        3: "Disgust",
        4: "Fear",
        5: "Happy",
        6: "Sadness",
        7: "Surprise",
    }
    return emotion.get(argument, "Invalid emotion")

def rolate(vector, angle):
    rx = vector[0]*(math.cos(angle)) - vector[1]*(math.sin(angle))
    ry = vector[0]*(math.sin(angle)) + vector[1]*(math.cos(angle))
    return [rx, ry]

#chuan hoa mang 68 diem landmark cua perk state theo neutral state
def normalize_perk_landmark(landmark_perk, landmark_neutral):
    neutral_center = landmark_neutral[30]
    perk_center = landmark_perk[30]
#     for lm in landmark_perk:
    #move perk landmark to match center of neutral landmark
    move_vector = [neutral_center[0] - perk_center[0], neutral_center[1] - perk_center[1]]
    for lm in landmark_perk:
        lm[0] += move_vector[0]
        lm[1] += move_vector[1]
    
    #scale the size of detected perk landmark based on neutrol landmark 27 to 30 (the nose)
    scale_neutral = [landmark_neutral[30][0] - landmark_neutral[27][0], landmark_neutral[30][1] - landmark_neutral[27][1]]
    scale_perk = [landmark_perk[30][0] - landmark_perk[27][0], landmark_perk[30][1] - landmark_perk[27][1]]
    ratio = math.sqrt(scale_neutral[0]*scale_neutral[0] + scale_neutral[1]*scale_neutral[1])/math.sqrt(scale_perk[0]*scale_perk[0]+scale_perk[1]*scale_perk[1]) 
    for lm in landmark_perk:
        lm[0] = (perk_center[0] - lm[0]) * (1 - ratio) + lm[0]
        lm[1] = (perk_center[1] - lm[1]) * (1 - ratio) + lm[1]
    
    #rolate the mask of perk to match neutral
    sign_y = scale_perk[0]*scale_neutral[1] - scale_perk[1]*scale_neutral[0]
    sign_x = scale_perk[0]*scale_neutral[0] + scale_perk[1]*scale_neutral[1]
    angle = math.atan2(sign_y, sign_x)
    for lm in landmark_perk:
        tmp_vector = [lm[0] - landmark_perk[30][0],lm[1] - landmark_perk[30][1]]
        new_vector = rolate(tmp_vector, angle)
        lm[0] = new_vector[0] + landmark_perk[30][0]
        lm[1] = new_vector[1] + landmark_perk[30][1]
        
    return landmark_perk


def coopNormalize(set1, set2):
    len1 = len(set1)
    len2 = len(set2)
    merged = set1 + set2
    normalized = normalize(merged, norm='max', axis=0)
    result = []
    for i in range(len1, len1 + len2):
        result.append(normalized[i])
    return result

def getEmotionPredict(neutral, perk):
	landmark_neutral = []
	landmark_perk = []
	landmarkChange = []

	for i in range (0,68):
		landmark_neutral.append([neutral[2*i+1], neutral[2*i]])
		landmark_perk.append([perk[2*i+1], perk[2*i]])

	landmark_perk = normalize_perk_landmark(landmark_perk, landmark_neutral)
	# models = Models.getAuModels()
	# facs = Models.getFacsList()
	# facs_predict = []
		
	# for model in models:
	# 	if(model.predict([landmarkChange]) >= 0.5):
	# 		facs_predict.append(facs[models.index(model)])
	# 		#print(facs[models.index(model)])


	# emotions = Models.emotions
	emotion_predict = {}
	# for emotion in emotions:
	# 	emotion_predict[emotion.name] = emotion.score(facs_predict)

	emotion_model = Models.getEmotionModel()
	normalizeData = Models.getNormalizeData()
	landmark = []
	for i in range(0,68):
		landmark.append(landmark_perk[i][0] - landmark_neutral[i][0])
		landmark.append(landmark_perk[i][1] - landmark_neutral[i][1])
	# print(coopNormalize(data_normalize, [landmark])[0])
	result = emotion_model.predict([coopNormalize(normalizeData, [landmark])[0]])
	#1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
	print(result)
	emotion_predict['emotion'] = emotion_decode(result[0])
	return emotion_predict