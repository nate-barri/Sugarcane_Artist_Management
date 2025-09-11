# Sugarcane Artist Management Platform

A comprehensive full-stack social media analytics dashboard for artists and content creators.

## Features

- **Multi-Platform Analytics**: Track performance across YouTube, Spotify, Instagram, TikTok, Facebook, and Twitter
- **Real-time Data Sync**: Automated data synchronization from social media platforms
- **Interactive Dashboard**: Beautiful, responsive dashboard with key metrics and charts
- **Report Generation**: Generate detailed PDF reports for performance analysis
- **User Authentication**: Secure JWT-based authentication system
- **Social Media Integration**: OAuth integration with major platforms
- **Background Tasks**: Celery-powered background data processing

## Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **SWR** - Data fetching and caching
- **Radix UI** - Accessible component primitives

### Backend
- **Django 4.2** - Python web framework
- **Django REST Framework** - API development
- **PostgreSQL** - Primary database
- **Redis** - Caching and task queue
- **Celery** - Background task processing
- **JWT Authentication** - Secure token-based auth

### Infrastructure
- **Docker** - Containerization
- **Nginx** - Reverse proxy and load balancer
- **Vercel** - Frontend deployment (optional)
- **Railway/Heroku** - Backend deployment (optional)

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+
- Docker & Docker Compose (for containerized setup)

### Local Development

1. **Clone the repository**
   \`\`\`bash
   git clone <repository-url>
   cd sugarcane-artist-management
   \`\`\`

2. **Setup Backend**
   \`\`\`bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
   # Setup database
   python manage.py migrate
   python ../scripts/setup_database.py
   
   # Start Django server
   python manage.py runserver 8000
   \`\`\`

3. **Setup Frontend**
   \`\`\`bash
   # In a new terminal, from project root
   npm install
   npm run dev
   \`\`\`

4. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/api
   - Django Admin: http://localhost:8000/admin

### Docker Setup

1. **Start all services**
   \`\`\`bash
   docker-compose up -d
   \`\`\`

2. **Access the Application**
   - Application: http://localhost
   - Direct Frontend: http://localhost:3000
   - Direct Backend: http://localhost:8000

### Quick Start Script

Use the provided script to start both frontend and backend:
\`\`\`bash
chmod +x scripts/start_fullstack.sh
./scripts/start_fullstack.sh
\`\`\`

## Demo Credentials

- **Email**: demo@sugarcane.com
- **Password**: demo123

## API Documentation

### Authentication Endpoints
- `POST /api/auth/login/` - User login
- `POST /api/auth/register/` - User registration
- `POST /api/auth/logout/` - User logout
- `POST /api/auth/refresh/` - Refresh JWT token
- `GET /api/auth/profile/` - Get user profile

### Analytics Endpoints
- `GET /api/analytics/dashboard/` - Dashboard overview data
- `GET /api/analytics/platform/{platform}/` - Platform-specific analytics
- `POST /api/analytics/sync/{platform}/` - Sync platform data

### Integration Endpoints
- `GET /api/integrations/` - List integrations
- `GET /api/integrations/status/` - Integration status summary
- `POST /api/integrations/connect/{platform}/` - Connect platform
- `POST /api/integrations/disconnect/{platform}/` - Disconnect platform

### Reports Endpoints
- `GET /api/reports/` - List reports
- `POST /api/reports/generate/` - Generate new report
- `GET /api/reports/download/{id}/` - Download report

## Environment Variables

### Backend (.env)
\`\`\`env
SECRET_KEY=your-secret-key
DEBUG=True
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
REDIS_URL=redis://localhost:6379/0

# Social Media API Keys
YOUTUBE_API_KEY=your-key
SPOTIFY_CLIENT_ID=your-id
SPOTIFY_CLIENT_SECRET=your-secret
# ... other platform credentials
\`\`\`

### Frontend (.env.local)
\`\`\`env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
NEXT_PUBLIC_APP_URL=http://localhost:3000
\`\`\`

## Social Media Integration Setup

### YouTube Data API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable YouTube Data API v3
4. Create credentials (API Key and OAuth 2.0)
5. Add credentials to backend environment variables

### Spotify Web API
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Get Client ID and Client Secret
4. Set redirect URI to your callback URL
5. Add credentials to backend environment variables

### Instagram Basic Display API
1. Go to [Facebook Developers](https://developers.facebook.com/)
2. Create a new app
3. Add Instagram Basic Display product
4. Configure OAuth redirect URIs
5. Add credentials to backend environment variables

### TikTok for Developers
1. Go to [TikTok Developers](https://developers.tiktok.com/)
2. Create a new app
3. Get Client Key and Client Secret
4. Configure redirect URIs
5. Add credentials to backend environment variables

## Deployment

### Frontend (Vercel)
1. Connect your GitHub repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

### Backend (Railway/Heroku)
1. Create a new project on Railway or Heroku
2. Connect your GitHub repository
3. Set environment variables
4. Add PostgreSQL and Redis add-ons
5. Deploy automatically on push to main branch

### Full Stack (Docker)
1. Update environment variables in docker-compose.prod.yml
2. Build and deploy:
   \`\`\`bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   \`\`\`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, email support@sugarcane.com or create an issue in the GitHub repository.
