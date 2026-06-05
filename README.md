# Run docker 
docker build -t credit-risk-app .    
docker run --name credit-risk-container -p 8000:8000 -p 7860:7860 credit-risk-app
docker run --rm --name credit-risk-container -p 8000:8000 -p 7860:7860 credit-risk-app