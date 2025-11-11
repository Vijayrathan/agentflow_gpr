#!/bin/bash

# Start script for GPRMax Chatbot
# This script starts both the Flask backend and React frontend

echo "🚀 Starting GPRMax Chatbot..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Please create one with your OPENAI_API_KEY."
fi

# Start Flask backend in background
echo "📡 Starting Flask backend on port 5002..."
python app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start React frontend
echo "🎨 Starting React frontend on port 3000..."
cd frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ Servers starting..."
echo "   Backend: http://localhost:5002"
echo "   Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait

