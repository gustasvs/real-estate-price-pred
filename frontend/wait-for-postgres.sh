#!/bin/sh
# wait-for-postgres.sh
# this script is needed because docker-compose depends_on does not wait for the postgres container to be ready

set -e

host="$1"
shift
cmd="$@"


until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

>&2 echo "Postgres is up - executing command"

echo "Running Prisma db push..."
npx prisma db push --schema=./prisma/schema.prisma --accept-data-loss

exec $cmd