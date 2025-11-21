#!/bin/bash

# Start script for gprMax Chatbot
# This script starts both the Flask backend and React frontend

echo "🚀 Starting gprMax Chatbot..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Please create one with your HF_TOKEN."
fi



# Start Flask backend in background
echo "📡 Starting Flask backend on port 5002..."
python app.py &
BACKEND_PID=$!

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:5002/api/health > /dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Backend may not be ready yet, but continuing..."
    fi
    sleep 1
done

# Start React frontend
echo "🎨 Starting React frontend on port 3000..."
cd frontend
export PORT=3000
# Don't set HOST - let React use default (localhost only, safer for school networks)
# Fix for webpack dev server allowedHosts issue
export DANGEROUSLY_DISABLE_HOST_CHECK=true
# Start npm in background and redirect output to log file
npm start > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for React to compile and start (can take 10-30 seconds)
echo "⏳ Waiting for React frontend to compile and start (this may take 15-30 seconds)..."
for i in {1..30}; do
    if lsof -i :3000 > /dev/null 2>&1; then
        echo "✅ React frontend is running on port 3000!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  React frontend is still starting (check frontend.log for progress)..."
        if [ -f frontend.log ]; then
            echo "   Last few lines of frontend.log:"
            tail -5 frontend.log | sed 's/^/   /'
        fi
    fi
    sleep 1
done

echo ""
echo "✅ Servers are running!"
echo ""
echo "🌐 Access the application:"
echo "   On the server: http://localhost:3000"
echo ""
echo "📡 For remote access (school network):"
echo "   Use SSH port forwarding from your local machine:"
echo "   ssh -L 3000:localhost:3000 -L 5002:localhost:5002 your_username@130.215.219.220"
echo "   Then access: http://localhost:3000 on your local machine"
echo ""
echo "📝 Note: The frontend will automatically proxy API requests to the backend."
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user interrupt
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait

