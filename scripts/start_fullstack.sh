# Full-stack startup script for Sugarcane Artist Management

echo "🎵 Starting Sugarcane Artist Management Full-Stack Application..."

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Start Django backend
echo "🔧 Starting Django Backend..."
if check_port 8000; then
    cd backend
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Start Django server in background
    python manage.py runserver 8000 &
    DJANGO_PID=$!
    echo "Django backend started with PID: $DJANGO_PID"
    cd ..
else
    echo "Django backend may already be running on port 8000"
fi

# Wait a moment for Django to start
sleep 3

# Start Next.js frontend
echo "🚀 Starting Next.js Frontend..."
if check_port 3000; then
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install
    fi
    
    # Start Next.js development server
    npm run dev &
    NEXTJS_PID=$!
    echo "Next.js frontend started with PID: $NEXTJS_PID"
else
    echo "Next.js frontend may already be running on port 3000"
fi

echo ""
echo "✅ Full-stack application is starting up!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000/api"
echo "👤 Django Admin: http://localhost:8000/admin"
echo ""
echo "Demo Credentials:"
echo "Email: demo@sugarcane.com"
echo "Password: demo123"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$DJANGO_PID" ]; then
        kill $DJANGO_PID 2>/dev/null
        echo "Django backend stopped"
    fi
    if [ ! -z "$NEXTJS_PID" ]; then
        kill $NEXTJS_PID 2>/dev/null
        echo "Next.js frontend stopped"
    fi
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup INT TERM

# Wait for user to stop the script
wait
