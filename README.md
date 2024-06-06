# Two-Stage-RecSys
A Two Stage Recommender System for Retail. (LFM, KNN, CatBoost and Other Heuristics)

## Prerequisites
Install docker and nvidia-container-runtime, then run in the shell:
```commandline
pip install requests streamlit
```

## Running:
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
     --mount type=bind,source="$(pwd)"/app/logs,target=/app/logs \
     -p 80:80 \
     recsys:latest
     
streamlit run ui.py 
```


## Load from image:
```commandline
docker load -i recsys_image.tar
```

