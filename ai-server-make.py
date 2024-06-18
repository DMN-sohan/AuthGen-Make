import cv2
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import bchlib
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from io import BytesIO
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BCH_POLYNOMIAL = 137
BCH_BITS = 5

sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())

model = tf.compat.v1.saved_model.load(sess, [tag_constants.SERVING], 'saved_models/stegastamp_pretrained')

input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
input_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(input_secret_name)
input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_name)

output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
output_stegastamp = tf.compat.v1.get_default_graph().get_tensor_by_name(output_stegastamp_name)
output_residual = tf.compat.v1.get_default_graph().get_tensor_by_name(output_residual_name)

width = 400
height = 400

bch = bchlib.BCH(prim_poly=BCH_POLYNOMIAL, t=BCH_BITS)

@app.post('/make_image')
async def make_image(file: UploadFile, secret: str = Form(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    image = Image.open(BytesIO(await file.read()))

    data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])

    size = (width, height)

    image = image.convert("RGB")
    image = np.array(ImageOps.fit(image, size), dtype=np.float32)
    image /= 255.

    feed_dict = {input_secret: [secret], input_image: [image]}

    hidden_img, residual = sess.run([output_stegastamp, output_residual], feed_dict=feed_dict)

    rescaled = (hidden_img[0] * 255).astype(np.uint8)

    im = Image.fromarray(np.array(rescaled))
    # im = ImageOps.fit(im, (1024, 1024))
    file_path = "out/out.png"
    im.save(file_path)

    return FileResponse(file_path, media_type='image/png', filename='out.png')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
