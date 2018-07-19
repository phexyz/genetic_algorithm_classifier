import requests

url = "https://www.kaggle.com/alxmamaev/flowers-recognition/downloads/flowers-recognition.zip/2"
r = requests.get(url, allow_redirects=True)
open('flowers-recognition.zip', 'wb').write(r.content)

url = "https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz"
r = requests.get(url, allow_redirects=True)
open("vgg16_weights.npz", "wb").write(r.content)