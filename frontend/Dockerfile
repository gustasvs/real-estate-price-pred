FROM node:22

WORKDIR /app

COPY package*.json ./

RUN npm install

RUN apt-get update && apt-get install -y postgresql-client

COPY . .

RUN npx prisma generate --schema=./prisma/schema.prisma
# RUN npx prisma db push
# RUN npx prisma migrate dev

# Expose the Next.js port
EXPOSE 3000

# Set environment variables for production
# ENV NODE_ENV production

# build for production
# RUN npm run build

# dev for development
# RUN npm run dev

# Start the Next.js app
# CMD ["npm", "run", "dev"]
COPY wait-for-postgres.sh /wait-for-postgres.sh
RUN chmod +x /wait-for-postgres.sh

# Start the application
CMD ["/wait-for-postgres.sh", "db", "npm run dev"]
