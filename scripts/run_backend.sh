# Sugarcane Artist Management Backend Startup Script

echo "🎵 Starting Sugarcane Artist Management Backend..."

# Navigate to backend directory
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run migrations
echo "Running database migrations..."
python manage.py makemigrations
python manage.py migrate

# Create superuser if it doesn't exist
echo "Creating superuser..."
python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(email='admin@sugarcane.com').exists():
    User.objects.create_superuser('admin', 'admin@sugarcane.com', 'admin123')
    print('Superuser created: admin@sugarcane.com / admin123')
else:
    print('Superuser already exists')
"

# Setup sample data
echo "Setting up sample data..."
python ../scripts/setup_database.py

# Start development server
echo "🚀 Starting Django development server..."
python manage.py runserver 8000
