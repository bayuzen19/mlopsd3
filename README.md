# Boston House Price Prediction 🏠

## Project Overview

This project demonstrates an end-to-end machine learning application for predicting house prices in Boston. It combines FastAPI for the backend service and Streamlit for an interactive frontend interface.


## Features
- 🔍 Interactive data exploration and visualization
- 📊 Comprehensive model analytics and performance metrics
- 🤖 Real-time price predictions using XGBoost
- 📈 Feature importance analysis
- 🎯 Model performance tracking
- 🖥️ User-friendly web interface

## Technology Stack
- **Backend**: FastAPI, Python 3.9
- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Containerization**: Docker
- **Version Control**: Git
- **Development**: VS Code

## Project Structure
```plaintext
boston_house_price/
│
├── .streamlit/                      # Streamlit configuration
│   └── config.toml                  # Streamlit settings
│
├── artifacts/                       # Model artifacts
│   ├── boston.csv                   # Dataset
│   ├── best_model.pkl              # Trained model
│   ├── scaler.pkl                  # Fitted scaler
│   └── metrics.json                # Model metrics
│
├── config/                          # Configuration
│   ├── __init__.py
│   └── config.py                   # Project configuration
│
├── logs/                           # Application logs
│   └── app.log
│
├── pages/                          # Streamlit pages
│   ├── 1_🏠_Overview.py            # Project overview
│   ├── 2_👤_Profile.py             # Developer profile
│   ├── 3_📊_Analytics.py           # Data analytics
│   └── 4_🔮_Predictions.py         # Price predictions
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── data_preparation.py         # Data preprocessing
│   ├── model.py                    # Model training
│   └── evaluation.py               # Model evaluation
│
├── static/                         # Static files
│   ├── css/
│   │   └── style.css              # Custom styling
│   └── img/
│       ├── profile.jpg            # Profile image
│       └── project.png            # Project diagram
│
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── logger.py                   # Logging setup
│   └── styling.py                  # Styling utilities
│
├── app.py                          # FastAPI application
├── Home.py                         # Streamlit main page
├── Dockerfile.fastapi              # FastAPI Dockerfile
├── Dockerfile.streamlit            # Streamlit Dockerfile
├── docker-compose.yml              # Docker composition
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

## Installation and Setup

### Prerequisites
- Python 3.9 or higher
- Docker and Docker Compose
- Git

### Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/bayuzen19/boston_house_price.git
cd boston_house_price
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows
.\venv\Scripts\activate
# For Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python train.py
```

5. Run the applications:
```bash
# Terminal 1 - Run FastAPI
uvicorn app:app --reload --port 8000

# Terminal 2 - Run Streamlit
streamlit run Home.py
```

### Docker Setup

#### Building Individual Images

1. FastAPI Image:
```bash
# Build FastAPI image
docker build -t boston-fastapi:latest -f Dockerfile.fastapi .

# Run FastAPI container
docker run -d -p 8000:8000 --name boston-fastapi boston-fastapi:latest
```

2. Streamlit Image:
```bash
# Build Streamlit image
docker build -t boston-streamlit:latest -f Dockerfile.streamlit .

# Run Streamlit container
docker run -d -p 8501:8501 --name boston-streamlit boston-streamlit:latest
```

3. Running with Network:
```bash
# Create network
docker network create boston-network

# Run FastAPI with network
docker run -d -p 8000:8000 --name boston-fastapi --network boston-network boston-fastapi:latest

# Run Streamlit with network
docker run -d -p 8501:8501 --name boston-streamlit --network boston-network boston-streamlit:latest
```
docker network create boston-network


Membuat jaringan Docker baru bernama "boston-network"
Memungkinkan container yang terhubung ke jaringan ini untuk berkomunikasi satu sama lain
Container dalam jaringan yang sama dapat saling merujuk menggunakan nama container mereka sebagai hostname


docker run -d -p 8000:8000 --name boston-fastapi --network boston-network boston-fastapi:latest


-d: Menjalankan container dalam mode detached (background)
-p 8000:8000: Memetakan port 8000 host ke port 8000 container
--name boston-fastapi: Memberikan nama "boston-fastapi" ke container
--network boston-network: Menghubungkan container ke jaringan "boston-network"
boston-fastapi:latest: Nama image yang akan dijalankan


docker run -d -p 8501:8501 --name boston-streamlit --network boston-network boston-streamlit:latest


-d: Menjalankan container dalam mode detached (background)
-p 8501:8501: Memetakan port 8501 host ke port 8501 container
--name boston-streamlit: Memberikan nama "boston-streamlit" ke container
--network boston-network: Menghubungkan container ke jaringan "boston-network"
boston-streamlit:latest: Nama image yang akan dijalankan

Manfaat menggunakan network:

Isolasi: Container dalam jaringan yang sama terisolasi dari container di jaringan lain
Komunikasi: Container dapat berkomunikasi menggunakan nama container sebagai hostname
Keamanan: Membatasi akses hanya ke container yang terhubung ke jaringan
DNS Internal: Docker menyediakan DNS internal untuk resolusi nama antar container

Contoh komunikasi:

Streamlit dapat mengakses FastAPI menggunakan URL http://boston-fastapi:8000
Tidak perlu menggunakan localhost atau IP address
Komunikasi lebih aman karena terisolasi dalam jaringan Docker

### Using Docker Compose

1. Build and run services:
```bash
# Build and run
docker-compose up --build

# Run in detached mode
docker-compose up -d --build
```

2. Stop services:
```bash
docker-compose down
```

## Usage Guide

### Web Interface
1. Access the Streamlit interface at `http://localhost:8501`
2. Navigate through the pages:
   - Overview: Project information and dataset exploration
   - Profile: Developer information
   - Analytics: Model performance and data analysis
   - Predictions: Make real-time predictions

### API Endpoints
Base URL: `http://localhost:8000`

1. Predict House Price:
```bash
POST /predict
Content-Type: application/json

{
    "LSTAT": 10.0,
    "RM": 6.0,
    "CRIM": 0.1,
    "PTRATIO": 15.0,
    "INDUS": 10.0,
    "TAX": 300.0,
    "NOX": 0.5,
    "B": 300.0
}
```

2. API Documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Model Information

### Dataset
- **Source**: Boston Housing Dataset
- **Features**: 8 key housing attributes
- **Target**: Median value of owner-occupied homes

### Model Details
- **Algorithm**: XGBoost Regressor
- **Preprocessing**: StandardScaler
- **Validation**: 5-fold cross-validation
- **Metrics**: R² Score, RMSE, MAE

## Docker Commands Reference

### Container Management
```bash
# List containers
docker ps

# Stop containers
docker stop boston-fastapi boston-streamlit

# Remove containers
docker rm boston-fastapi boston-streamlit
```

### Image Management
```bash
# List images
docker images

# Remove images
docker rmi boston-fastapi:latest boston-streamlit:latest
```

### Logs and Debugging
```bash
# View logs
docker logs boston-fastapi
docker logs boston-streamlit

# Follow logs
docker logs -f boston-fastapi
```

## Troubleshooting

### Common Issues

1. Port Conflicts
```bash
# Check port usage
lsof -i :8000
lsof -i :8501

# Use alternative ports
docker run -d -p 8001:8000 boston-fastapi:latest
```

2. Permission Issues
```bash
# Run with sudo (Linux)
sudo docker-compose up

# Add user to docker group
sudo usermod -aG docker $USER
```

3. Memory Issues
```bash
# View resource usage
docker stats

# Set memory limits
docker run -d -p 8000:8000 --memory=1g boston-fastapi:latest
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Development Team
- **Bayuzen Ahmad** - Data Scientist
  - Email: bayuzen19@gmail.com
  - LinkedIn: [Bayuzen Ahmad](https://www.linkedin.com/in/bayuzenahmad/)
  - GitHub: [bayuzen19](https://github.com/bayuzen19)

## Acknowledgments
- Boston Housing Dataset contributors
- Streamlit and FastAPI communities
- XGBoost development team

---
📫 For support, email bayuzen19@gmail.com or create an issue in the repository.

Built with ❤️ using Python, FastAPI, and Streamlit
