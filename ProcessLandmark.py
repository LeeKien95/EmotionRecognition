import Models
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
    return emotion.get(argument, "Invalid month")

def coopNormalize(set1, set2):
    len1 = len(set1)
    len2 = len(set2)
    merged = set1 + set2
    normalized = normalize(merged, norm='max', axis=0)
    result = []
    for i in range(len1, len1 + len2):
        result.append(normalized[i])
    return result

def getEmotionPredict(landmarkChange):
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
	result = emotion_model.predict([coopNormalize(normalizeData, [landmarkChange])[0]])
	#1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
	print(result)
	emotion_predict['emotion'] = emotion_decode(result[0])
	return emotion_predict