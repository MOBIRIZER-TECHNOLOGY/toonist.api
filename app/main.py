# app/main.py

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse
import os
from os import getcwd
import cv2 as cv
import uuid
import time
from pathlib import Path
from starlette.staticfiles import StaticFiles
from os.path import dirname, abspath, join

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

app = FastAPI(title="FastAPI, Docker, and Traefik")
app.mount("/static", StaticFiles(directory="static"), name="static")

input_dir_path = join(getcwd(), 'static/input/')
output_dir_path = join(getcwd(), 'static/output/')

print("pawan input_dir_path",input_dir_path)

cartoon_anime_model = pipeline(Tasks.image_portrait_stylization,model= join(getcwd(), 'damo/cv_unet_person-image-cartoon_compound-models'))
cartoon_3d_model = pipeline(Tasks.image_portrait_stylization,model=join(getcwd(), 'damo/cv_unet_person-image-cartoon-3d_compound-models'))
cartoon_handdrawn_model = pipeline(Tasks.image_portrait_stylization,model=join(getcwd(), 'damo/cv_unet_person-image-cartoon-handdrawn_compound-models'))
cartoon_sketch_model = pipeline(Tasks.image_portrait_stylization,model=join(getcwd(), 'damo/cv_unet_person-image-cartoon-sketch_compound-models'))
cartoon_artstyle_model = pipeline(Tasks.image_portrait_stylization,model=join(getcwd(), 'damo/cv_unet_person-image-cartoon-artstyle_compound-models'))

@app.post("/predict/image")
async def predict_api(request: Request, style: str = "3d",image: UploadFile = File(...)):
    extension = image.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    suffix = Path(image.filename).suffix
    filename = time.strftime(str(uuid.uuid4().hex) + "%Y%m%d-%H%M%S" + suffix )
    
    
    input_file_name= input_dir_path + filename

    with open(input_file_name,'wb+') as f:
        f.write(image.file.read())
        f.close()

    img_cartoon = cartoon_anime_model

    if style == "anime":
        img_cartoon = cartoon_anime_model
    elif style == "3d":
        img_cartoon = cartoon_3d_model
    elif style == "handdrawn":
        img_cartoon = img_cartoon
    elif style == "sketch":
        img_cartoon = img_cartoon
    elif style == "artstyle":
        img_cartoon = img_cartoon

    result = img_cartoon(input_file_name)

    output_file_name = output_dir_path + filename

    cv.imwrite(output_file_name,result[OutputKeys.OUTPUT_IMG])

    # file_url = request.client.host +  "/file/" +  filename
    file_url = "http://0.0.0.0:8081" +  "/file/" +  filename

    print("output_file_name:--",file_url)
    data={"style":style,"image_url":file_url}
    return data

@app.get("/file/{name_file}")
def get_file(name_file: str):
    output_file_name = output_dir_path + name_file
    return FileResponse(path=output_file_name)

@app.get("/")
def read_root():
    return {"hello": "world"}