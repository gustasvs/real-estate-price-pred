import * as Minio from 'minio';

// process.env.MINIO_HOST || 'localhost'
const minioClient = new Minio.Client({
  endPoint: 'minio',
  // port: parseInt(process.env.MINIO_PORT || '9000'),
  port: 9000,
  useSSL: false,
  accessKey: process.env.MINIO_ROOT_USER || 'minioadmin',
  secretKey: process.env.MINIO_ROOT_PASSWORD || 'minioadmin',
});

export default minioClient;