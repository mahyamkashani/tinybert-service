# End-to-End ML Deployment

## Docker Setup

### Build Docker Image

```bash
docker build -t flowtale .
```

### Run Docker Container

```bash
docker run -d -p 8000:8000 --name flowtale-app flowtale
```

This will:
- Build the Docker image with the tag `flowtale`
- Run the container in detached mode (-d)
- Map port 8000 from the container to port 8000 on your host
- Name the container `flowtale-app`

### Access the Application

Once running, you can access:
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Stop and Remove Container

```bash
docker stop flowtale-app
docker rm flowtale-app
```

## Deploy TinyBERT model using AWS

Client (Postman / Web / App)
        |
        v
   EC2 Instance
  (FastAPI + TinyBERT)
        |
        v
     S3 Bucket
 (stores fine-tuned model)

```
Client → Nginx → Gunicorn (with Uvicorn workers) → FastAPI app

```