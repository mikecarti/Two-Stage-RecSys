# Two-Stage-RecSys
A Two Stage Recommender System for Retail. (LFM, KNN, CatBoost and Other Heuristics)


## Running:
Install docker and nvidia-container-runtime, then run in the shell:

```commandline
# opional
docker stop recsys
docker rm recsys
```

```commandline
docker build -t recsys .
docker run -it \
     --name recsys \
     --mount type=bind,source="$(pwd)"/app/model_files,target=/app/model_files \
     -p 80:80 \
     recsys:latest

docker exec -it recsys /bin/bash
```

## Restart 
```commandline
docker stop recsys && docker rm recsys && docker build . -t recsys && docker run --name recsys -p 80:80 recsys
```

## Run from image:
```commandline
docker load -i recsys_image.tar
docker run --name recsys -p 80:80 recsys
```

