from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

try:
    scaled = joblib.load('scaled_pry.pkl')
    pca = joblib.load('pca_pry.pkl')
    modelo = joblib.load('RandomForestClassifier_pry.pkl')
except Exception as e:
    app.logger.error(f'Error al cargar los modelos: {str(e)}')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        localizacion_inicial = request.form['localizacion_inicial']
        aspecto_topografico = request.form['aspecto_topografico']
        numero_zonas_afectadas = request.form['numero_zonas_afectadas']
        isquemia = request.form['isquemia']
        infeccion = request.form['infeccion']
        edema = request.form['edema']
        neuropatia = request.form['neuropatia']
        profundidad = request.form['profundidad']
        area = request.form['area']
        fase_cicatrizacion = request.form['fase_cicatrizacion']

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[localizacion_inicial, aspecto_topografico, numero_zonas_afectadas, isquemia, infeccion, edema, neuropatia, profundidad, area, fase_cicatrizacion]], 
                               columns=['localizacion_inicial', 'aspecto_topografico', 'numero_zonas_afectadas', 'isquemia', 'infeccion', 'edema', 'neuropatia', 'profundidad', 'area', 'fase_cicatrizacion'])
        data_df_scaled = scaled.transform(data_df)
        
        data_for_pca = pd.DataFrame(data_df_scaled, columns=data_df.columns)[['localizacion_inicial', 'isquemia', 'infeccion', 'edema', 'neuropatia', 'profundidad']]
        data_df_pca = pca.transform(data_for_pca)
        
        # Realizar predicciones
        prediction = modelo.predict(data_df_pca)
        app.logger.debug(f'Predicción: {prediction}')
        
        respuesta = int(prediction[0]) 

        if respuesta == 0:
            severidad = "Leve"
        elif respuesta == 1:
            severidad = "Moderado"
        else:
            severidad = "Grave"
        
        return jsonify({'herida': severidad})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
