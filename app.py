from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np

app = Flask(__name__)

#load model
	model = pickle.load(open('knn.pkl', 'rb'))

@app.route('/', methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		umur = float(request.form ['umur']) 
		sakit_kepala = float(request.form ['sakit_kepala']) 
		anosmia = float(request.form ['anosmia']) 
		demam = float(request.form ['demam'])
		batuk = float(request.form ['batuk'])
		kehilangan_nafsu_makan =  float(request.form ['kehilangan_nafsu_makan'])
		suara_serak = float(request.form ['suara_serak'])
		sakit_tenggorokan = float(request.form ['sakit_tenggorokan'])
		nyeri_dada = float(request.form ['nyeri_dada'])
		lemas = float(request.form ['lemas'])
		kebingungan = float(request.form ['kebingungan'])
		nyeri_otot = float(request.form ['nyeri_otot'])
		sesak_nafas = float(request.form ['sesak_nafas'])
		diare = float(request.form ['diare'])
		sakit_perut = float(request.form ['sakit_perut'])
		tanpa_gejala = float(request.form ['tanpa_gejala'])
		komorbid = float(request.form ['komorbid'])

		datas = np.array((umur, sakit_kepala, anosmia, demam, batuk, kehilangan_nafsu_makan, suara_serak, sakit_tenggorokan, nyeri_dada, lemas, kebingungan, nyeri_otot, sesak_nafas, diare, sakit_perut, tanpa_gejala, komorbid))
		datas = np.reshape(datas, (1, -1))

		isCovid = model.predict(datas)
		
		return render_template('hasil.html')

	else:
		return render_template('index.html')

if __name__ == "__main__":
	app.run(debug=True)