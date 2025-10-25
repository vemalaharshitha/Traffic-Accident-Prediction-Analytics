# 🚦 Traffic Accident Prediction & Analytics Dashboard

A **smart traffic analytics system** that uses **YOLOv8** for real-time vehicle detection and **accident risk prediction**.  
Users can **upload traffic videos**, view **instant analytics** on a modern dashboard, and **download the processed output** with risk detection overlays.

---

## ✨ Key Features

- 📥 Upload any `.mp4` traffic video directly from the dashboard  
- 🤖 Real-time vehicle detection powered by YOLOv8  
- ⚡ Automatic accident risk level calculation (Low / Medium / High)  
- 📊 Live analytics including:  
  - Vehicle counts  
  - Risk level & probability  
  - Processing status & alerts  
- 💾 Processed video saved in the `processed/` folder automatically  
- ⬇️ Download processed video output directly from the dashboard  
- 🧠 No webcam required (supports file uploads)

---

## 🏗️ Tech Stack

- **Frontend:** HTML5, Tailwind CSS, JavaScript  
- **Backend:** Python, Flask  
- **Model:** YOLOv8 (Ultralytics)  
- **Computer Vision:** OpenCV  
- **Data Processing:** NumPy

---

## 🧰 Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/traffic-accident-prediction.git
cd traffic-accident-prediction
🧠 How It Works

User uploads a traffic video through the dashboard.

1.The backend uses YOLOv8 to detect cars, buses, trucks, and bikes.
2.Risk scores are calculated dynamically based on vehicle density.
3.Live analytics are displayed on the dashboard during processing.
4.The processed video with detection overlays is saved in processed/.
5.The user can download the analyzed video directly from the U

📜 License
This project is licensed under the MIT License — free to use, share, and modify.

🚀 Future Enhancements

🗺️ Live accident hotspot map visualization

📈 Risk probability timeline charts

🌦️ Integration of weather data for risk adjustment

📨 Alert system (Email / SMS) for high-risk detection

☁️ Cloud deployment (AWS / Azure) for real-time monitoring
