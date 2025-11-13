# Use Node.js 20 with Python support
FROM node:20-bullseye

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node dependencies
RUN npm ci

# Copy Python requirements and install
COPY scripts/requirements.txt ./scripts/
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r scripts/requirements.txt

# Copy all files
COPY . .

ENV DATABASE_URL="postgresql://neondb_owner:npg_dGzvq4CJPRx7@ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech:5432/neondb"
ENV SKIP_ENV_VALIDATION=1

# Build Next.js app
RUN npm run build

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]
