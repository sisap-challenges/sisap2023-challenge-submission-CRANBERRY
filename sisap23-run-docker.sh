docker build --no-cache -t sisap23/cranberry .
docker run -v /home/sisap23evaluation/data:/data:ro -v ./result:/result -it sisap23/cranberry 300K
