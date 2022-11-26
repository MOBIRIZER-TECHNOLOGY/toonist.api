# cdproject

This project was generated via [manage-fastapi](https://ycd.github.io/manage-fastapi/)! :tada:

## License

This project is licensed under the terms of the None license.
pip install pipreqs

docker-compose build
docker-compose up -d
docker-compose logs -f
docker system prune

http://localhost:8008

curl -H Host:fastapi.localhost http://0.0.0.0:8008

uvicorn app.main:app --host 0.0.0.0 --port 8081 --reload

http://0.0.0.0:8081/

http://0.0.0.0:8081/file/6e46b040751c4944a887d3b6a71a4e6220221125-001710.jpeg


cartoon-api-container |    

 cartoon_anime_model = pipeline(Tasks.image_portrait_stylization,model='damo/cv_unet_person-image-cartoon_compound-models')

<!-- 

FROM continuumio/miniconda3

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the environment, and make sure it's activated:
RUN conda activate fastApiDocker

WORKDIR /app

VOLUME ["/damo"]

# copy project
COPY . .




 -->