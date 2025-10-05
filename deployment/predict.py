"""Web service para prediccion de la calidad del vino

Flask API para que se encarga de responder la calidad del vino con base en el modelo de 
prediccion, el servicio expone el endpoint /predict para recibir las
carateristicas del vino y devolver la calidad

Author: Especializacion UdM Equipos de Trabajo
Version: 1.0
"""

import pickle
import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and DictVectorizer at application startup
try:
    with open('../src/app/train/wine-quality.bin', 'rb') as f_in:
        logger.info('🔄 Loading model and DictVectorizer...')
        (dv, model) = pickle.load(f_in)
        logger.info('✅ Model and DV loaded successfully')
except FileNotFoundError:
    logger.error('❌ Error: wine-quality.bin file not found')
    raise
except Exception as e:
    logger.error(f'❌ Error loading model: {e}')
    raise


def prepare_features(wine_features):
    """
    Prepare features needed for prediction from wine quality data.
    
    Args:
        wine_features (dict): Dictionary with wine data that must contain:
            - volatile_acidity (float): nivel de acidez
            - residual_sugar (float): azucar residual
            - density (float): medida de densidad
            - alcohol (float): nivel de alcohol
    
    Returns:
        dict: Dictionary with processed features:
            - quality (int): Nivel de calidad
    
    Example:
        >>> wine_features = {
        ...     'volatile_acidity': 0.3,
        ...     'residual_sugar': 1.6,
        ...     'density': 0.99534,
        ...     'alcohol': 8.5
        ... }
        >>> features = prepare_features(wine_features)
        >>> print(features)
        {'wine_quality': '6'}
    """
    features = {}
    features['wine_quality'] = '%s_%s' % (wine_features['volatile_acidity'], wine_features['residual_sugar'], wine_features['density'], wine_features['alcohol'])
    logger.info(f"✅ Features prepared: volatile_acidity={wine_features['volatile_acidity']}, residual_sugar={wine_features['residual_sugar']}, density={wine_features['density']}, alcohol={wine_features['alcohol']}")
    return features


def predict(features):
    """
    Perform quality prediction using the loaded model.
    
    Args:
        features (dict): Features prepared with prepare_features()
    
    Returns:
        int: Predicted quality level
    
    Note:
        - Uses DictVectorizer to transform categorical features
        - Applies pre-trained linear regression model
        - Returns prediction as float for JSON serialization
    
    Example:
        >>> wine_features = {'wine_quality': '6'}
        >>> wine_quality = predict(features)
        >>> print(f"Predicted quality: {quality:.2f} level")
    """
    X = dv.transform(features)
    preds = model.predict(X)
    predicted_quality = float(preds[0])
    logger.info(f"🎯 Prediction made: {predicted_quality:.2f} level quality")
    return predicted_quality


# Create Flask application
app = Flask('quality-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    REST endpoint for wine quality prediction.
    
    Method: POST
    Content-Type: application/json
    
    Request Body:
        {
            "volatile_acidity": float,
            "residual_sugar": float,
            "density": float,
            "alcohol": float
        }
    
    Response:
        {
            "quality": int  # Predicted quality level
        }
    
    Returns:
        JSON response with predicted quality or 400/500 error
    
    Example:
        curl -X POST http://localhost:9696/predict \
             -H "Content-Type: application/json" \
             -d '{"volatile_acidity": 1.2, "residual_sugar": 8.4, "density": 0.99945, "alcohol": 11.2}'
        
        Response: {"quality": 4}
    """
    try:
        # Get JSON data from request
        wine_features = request.get_json()
        
        if not wine_features:
            logger.error("❌ Request without JSON data")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['volatile_acidity', 'residual_sugar', 'density', 'alcohol']
        for field in required_fields:
            if field not in wine_features:
                logger.error(f"❌ Missing required field: {field}")
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        logger.info(f"New prediction: {wine_features['volatile_acidity']} -> {wine_features['residual_sugar']} -> {wine_features['density']} -> {wine_features['alcohol']}  ")
        
        # Prepare features and predict
        features = prepare_features(wine_features)
        pred = predict(features)
        
        result = {
            'quality': pred,
            'volatile_acidity': wine_features['volatile_acidity'],
            'residual_sugar': wine_features['residual_sugar'],
            'density': wine_features['density'],
            'alcohol': wine_features['alcohol']
        }
        
        logger.info(f"✅ Response sent: {pred:.2f} level")
        return jsonify(result)
        
    except KeyError as e:
        logger.error(f"❌ Missing field in request: {e}")
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"❌ Error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns:
        JSON response with service status
    
    Example:
        curl http://localhost:9696/health
        
        Response: {"status": "healthy", "model_loaded": true}
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'dv_loaded': dv is not None,
        'service': 'Wine quality Prediction'
    })


if __name__ == "__main__":
    """
    Main entry point to run the Flask server.
    
    Configuration:
        - Debug: True (development only)
        - Host: 0.0.0.0 (accepts external connections)
        - Port: 9696
    """
    logger.info("🚀 Starting Flask server on port 9696...")
    app.run(debug=True, host='0.0.0.0', port=9696)
