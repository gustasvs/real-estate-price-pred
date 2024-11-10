import { createClient } from 'redis';

const redisClient = createClient();
redisClient.connect();

redisClient.on('error', (err) => console.log('Redis Client Error', err));

export default redisClient;