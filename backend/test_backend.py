
import requests

# URL of the API
url = 'http://127.0.0.1:8000/predict'

# Create a dummy image for testing if we don't have one, or use a real one
# Let's creating a simple blank image in memory
from PIL import Image
import io

img = Image.new('RGB', (224, 224), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr = img_byte_arr.getvalue()

files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
