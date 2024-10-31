FROM node:22

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

# optionally run these to initialise the database
# RUN npx prisma generate
# RUN npx prisma db push

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
