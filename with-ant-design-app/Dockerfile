# Use the official Node.js image as the base
FROM node:22

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Install Prisma CLI globally
RUN npm install prisma --save-dev

# Copy the rest of the app's source code
COPY . .

# Run Prisma generate to create Prisma Client
RUN npx prisma generate

# Expose the Next.js port
EXPOSE 3000

# Set environment variables for production
# ENV NODE_ENV production

# build for production
# RUN npm run build

# dev for development
# RUN npm run dev

# Start the Next.js app
CMD ["npm", "run", "dev"]
