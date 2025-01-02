import amqp from 'amqplib';

async function sendRabbitMessage(messageObject: string) {
  try {
    amqp.connect('amqp://localhost').then(async (connection) => {
        console.log("Connected to RabbitMQ");
        const channel = await connection.createChannel();
        const exchange = 'objectCreationExchange';
        const queue = 'objectCreationQueue';
        const routingKey = 'objectCreationRoutingKey';
    
        await channel.assertExchange(exchange, 'direct', { durable: true });
        await channel.assertQueue(queue, { durable: true });
        await channel.bindQueue(queue, exchange, routingKey);
    
        channel.publish(exchange, routingKey, Buffer.from(JSON.stringify({ objectId: messageObject })));
      });
    console.log("RabbitMQ connection closed");
    return { success: true };
  } catch (error) {
    console.error("Error connecting to RabbitMQ:", error);
    return { success: false, error };
  }
}

export default sendRabbitMessage;
