FROM node:22

WORKDIR /app

COPY package*.json ./

RUN npm install

RUN apt-get update && apt-get install -y postgresql-client

COPY . /app

RUN npx prisma generate --schema=./prisma/schema.prisma
# RUN npx prisma db push
# RUN npx prisma migrate dev

# Expose the Next.js port
EXPOSE 3000

COPY wait-for-postgres.sh /wait-for-postgres.sh
RUN chmod +x /wait-for-postgres.sh


# build && npm run start

CMD ["/bin/sh", "-c", "/wait-for-postgres.sh db && npm run dev"]
