# Machine Learning FastAPI back-end

## Deploy Using Docker

### Prerequisites

Docker Desktop installed on your machine  
👉 https://docs.docker.com/desktop/

---

### Steps

1. Copy the /deploy folder from this repository to your local machine.
2. Open a terminal and navigate into the copied folder:
```shell
cd path/to/deploy
```
3. Start the container:
```shell
docker compose up -d
```
4. Open the application in your browser:  
👉 http://localhost:8002/

The[web UI](https://github.com/lendoo73/ml-react) is served by FastAPI from the /build directory  

The API docs are available at:  
👉 http://localhost:8002/api/docs  
👉 http://localhost:8002/api/redoc  