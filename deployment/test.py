"""NYC Taxi quality Prediction - Client Test Script

Test script for sending HTTP requests to the prediction service.
Useful for validating that the REST API works correctly.

Author: Especializacion UdeM Equipos de Trabajo
Version: 1.0
"""

import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_prediction_api(base_url='http://localhost:9696'):
    """
    Test the API prediction endpoint.
    
    Args:
        base_url (str): Base URL of the service (default: http://localhost:9696)
    
    Returns:
        dict: API response with the prediction
    
    Example:
        >>> result = test_prediction_api()
        >>> print(f"Predicted quality: {result['quality']:.2f} level")
    """
    # Test data for a typical wine
    wine = {
        "volatile_acidity": 11.2, 
        "residual_sugar": 1.6,   
        "density": 0.99114,   # Distance in miles
        "alcohol": 11.2
    }
    
    url = f'{base_url}/predict'
    
    try:
        logger.info(f"🚕 Sending test request to {url}")
        logger.info(f"📊 wine data: {json.dumps(wine, indent=2)}")
        
        # Send POST request
        response = requests.post(url, json=wine, timeout=10)
        
        # Check status code
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Request successful!")
            logger.info(f"📈 Predicted quality: {result.get('quality', 'N/A'):.2f} level")
            
            # Show complete response if it includes more fields
            if len(result) > 1:
                logger.info("📋 Complete response:")
                for key, value in result.items():
                    logger.info(f"   {key}: {value}")
            
            return result
        else:
            logger.error(f"❌ Error HTTP {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Error: Could not connect to server")
        logger.error("   Make sure the service is running on port 9696")
        return None
    except requests.exceptions.Timeout:
        logger.error("❌ Error: Connection timeout")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return None


def test_health_endpoint(base_url='http://localhost:9696'):
    """
    Test the health check endpoint.
    
    Args:
        base_url (str): Base URL of the service
    
    Returns:
        dict: Service status
    """
    url = f'{base_url}/health'
    
    try:
        logger.info(f"🏥 Checking health endpoint at {url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            logger.info("✅ Service healthy!")
            logger.info(f"📊 Status: {health_data}")
            return health_data
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error in health check: {e}")
        return None


def run_comprehensive_test():
    """
    Run a comprehensive test suite for the API.
    
    Includes:
    - Health check
    - Basic prediction
    - Edge cases (long/Vino baratos)
    """
    logger.info("🧪 Starting comprehensive test suite...")
    
    base_url = 'http://localhost:9696'
    
    # 1. Health Check
    logger.info("\n1️⃣ Testing Health Check...")
    health_result = test_health_endpoint(base_url)
    
    if not health_result:
        logger.error("❌ Health check failed, aborting tests")
        return
    
    # 2. Basic prediction
    logger.info("\n2️⃣ Testing basic prediction...")
    basic_result = test_prediction_api(base_url)
    
    # 3. Edge cases
    logger.info("\n3️⃣ Testing edge cases...")
    
    # Vino barato
    cheap_wine = {
        "volatile_acidity": 0.32,  
        "residual_sugar": 8,  
        "density": 0.99490,
        "alcohol": 9.6
    }
    
    # Vino costoso
    expensive_wine = {
        "volatile_acidity": 0.29,    
        "residual_sugar": 1.1,  
        "density": 0.98869,
        "alcohol": 12.8
    }
    
    test_cases = [
        ("Vino barato", cheap_wine),
        ("Vino costoso", expensive_wine)
    ]
    
    for case_name, trip_data in test_cases:
        logger.info(f"\n   🔍 Case: {case_name}")
        try:
            url = f'{base_url}/predict'
            response = requests.post(url, json=trip_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                quality = result.get('quality', 0)
                logger.info(f"   ✅ {case_name}: {quality:.2f} level")
            else:
                logger.error(f"   ❌ {case_name} failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"   ❌ Error in {case_name}: {e}")
    
    logger.info("\n🎉 Test suite completed!")


if __name__ == "__main__":
    """
    Run tests when the script is executed directly.
    """
    logger.info("🚀 Starting test quality wine prediction API...")
    
    # Run comprehensive test suite
    run_comprehensive_test()