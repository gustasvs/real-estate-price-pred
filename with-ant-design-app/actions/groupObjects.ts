"use server";

import { useSession } from "next-auth/react";
import { db } from "../db";
import { auth } from "../auth";

import amqp from 'amqplib'
import { revalidatePath } from "next/cache";
import { generateDownloadUrl } from "../app/api/generateDownloadUrl";


export const getObject = async (objectId: string) => {

  const session = await auth();

  if (!session) {
    return { error: "User not authenticated" };
  }

  amqp.connect('amqp://localhost').then(async (connection) => {
    console.log("Connected to RabbitMQ");
    const channel = await connection.createChannel();
    const exchange = 'objectCreationExchange';
    const queue = 'objectCreationQueue';
    const routingKey = 'objectCreationRoutingKey';

    await channel.assertExchange(exchange, 'direct', { durable: true });
    await channel.assertQueue(queue, { durable: true });
    await channel.bindQueue(queue, exchange, routingKey);

    channel.publish(exchange, routingKey, Buffer.from(JSON.stringify({ objectId })));
  });

  try {
    const object = await db.residence.findUnique({
      where: { id: objectId },
    });
    return object;
  } catch (error) {
    console.error("Error fetching object:", error);
    return { error: "Failed to get object" };
  }
};

export const getObjects = async (groupId: string, filter: any) => {

  // 2 second delay
  // await new Promise((resolve) => setTimeout(resolve, 2000));

  console.log("Filter:", filter);

  const session = await auth();
  if (!session) {
    return { error: "User not authenticated" };
  }

  const user = session?.user;

  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }

  const userId = user.id;

  try {

    const residenceName = filter?.residenceName || "";

    const objects = await db.residence.findMany({
      where: { 
        groupId: groupId, 
        // residenceGroup: { userId: userId },
        ... (residenceName ? { name: { contains: residenceName, mode: "insensitive" } } : {}),
      },

    });

    const objectsWithPresignedDownloadUrls = await Promise.all(
      objects.map(async (obj) => {
        // console.log("obj", obj);
        // return obj;
        if (obj && obj.pictures && Array.isArray(obj.pictures)) {
          const updatedPictures = await Promise.all(
            obj.pictures.map(async (picture) => {
              const downloadUrl = await generateDownloadUrl(picture, 'object-pictures');
              return {
                fileName: picture,
                downloadUrl: typeof downloadUrl === 'object' && 'error' in downloadUrl ? null : downloadUrl,
              };
            })
          );
          return { ...obj, pictures: updatedPictures };
        }
        return obj;
      })
    );

    return objectsWithPresignedDownloadUrls;
  } catch (error) {
    console.error("Error fetching objects:", error);
    return { error: "Failed to get objects" };
  }
};

export const getMyFavoriteObjects = async () => {
  const session = await auth();
  if (!session) {
    return { error: "User not authenticated" };
  }

  if (!session.user || !session.user.id) {
    return { error: "Unauthorized" };
  }


  // TODO add userId to residence object
  try {
    const objects = await db.residence.findMany({
      where: {
        favourite: true,
        residenceGroup: {
          userId: session.user.id,
        },
      },
    });
    return objects;
  } catch (error) {
    console.error("Error fetching favorite objects:", error);
    return { error: "Failed to get favorite objects" };
  }
};


// model Residence {
//   id          String @id @default(uuid())
//   name        String
//   address     String
//   area        Float
//   description String

//   bedroomCount  Int
//   bathroomCount Int
//   parkingCount  Int

//   price          Float
//   predictedPrice Float

//   groupId        String
//   residenceGroup ResidenceGroup @relation(fields: [groupId], references: [id], onDelete: Cascade)

//   pictures String[]

//   createdAt DateTime @default(now())
//   updatedAt DateTime @updatedAt
// }

export const createObject = async (
  groupId: string,
  objectData: {
    name: string;
    address: string;
    area: number;
    description: string;
    bedroomCount: number;
    bathroomCount: number;
    parkingCount: number;
    price: number;
    pictures: string[];
  }
) => {
  try {

    
  const session = await auth();

  const user = session?.user;

  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }

  const userId = user.id;

  const groupExists = await db.residenceGroup.findUnique({
    where: { id: groupId },
  });
  
  if (!groupExists) {
    return { error: "Invalid groupId" };
  }

  console.log("objectData", objectData);


    const picturesExtracted = objectData.pictures.map((picture: string | { pictureUrl: string }) => {
      if (typeof picture === "string") {
        return picture;
      }
      if (typeof picture === "object") {
        return picture.pictureUrl;
      }
      return picture;
    });

    const newObject = await db.residence.create({
      data: {
      name: objectData.name,
      address: objectData.address,
      description: objectData.description,
      pictures: picturesExtracted,
      groupId: groupId,
      area: parseFloat(objectData.area.toString()),
      bedroomCount: parseInt(objectData.bedroomCount.toString()),
      bathroomCount: parseInt(objectData.bathroomCount.toString()),
      parkingCount: objectData.parkingCount ? 1 : 0,
      price: parseFloat(objectData.price.toString()),
      predictedPrice: 0,
      },
    });

    try {
  
    } catch (error) {
      console.error("Error queuing object rerendering:", error);
    }

    return newObject;
  } catch (error) {
    console.error("Error creating object:", error);
    return { error: "Failed to create object" };
  }
};

export const updateObject = async (
  objectId: string,
  objectData: {
    name?: string;
    address?: string;
    area?: number;
    description?: string;
    bedroomCount?: number;
    bathroomCount?: number;
    parkingCount?: number;
    price?: number;
    predictedPrice?: number;
    groupId?: string;
    pictures?: { pictureUrl: string; status: string }[];
    favourite?: boolean;
  }
) => {
  try {
    // Destructure and filter out undefined/null values
    const { pictures, ...dataWithoutPictures } = objectData;

    const validData = Object.fromEntries(
      Object.entries(dataWithoutPictures).filter(([_, value]) => value != null)
    );
    // Update non-picture fields
    const updatedObject = await db.residence.update({
      where: { id: objectId },
      data: validData,
    });

    const currentPictures = updatedObject.pictures;

    console.log("Current pictures:", currentPictures);
    console.log("Passed pictures:", pictures);

    // Handle picture updates if provided
    if (pictures && Array.isArray(pictures)) {

      // pictures that are in current pictures but not in the new pictures
      // this is just for logging / auditting purposes
      const deletedPictures = currentPictures.filter(
        (existingUrl) =>
          !pictures.some((newPicture) => newPicture.pictureUrl === existingUrl)
      );
      console.log("Pictures to delete:", deletedPictures);

      // Compute new pictures array
      const newPictures = pictures.map((picture) => picture.pictureUrl);

        console.log("New pictures array:", newPictures);

        // Update pictures
        await db.residence.update({
          where: { id: objectId },
          data: { pictures: newPictures },
        });
    }

    return { success: true, updatedObject };
  } catch (error) {
    console.error("Error updating object:", error);
    return { error: "Failed to update object" };
  }
};


export const deleteObject = async (objectId: string) => {
  try {
    await db.residence.delete({
      where: { id: objectId },
    });
    return { success: true };
  } catch (error) {
    console.error("Error deleting object:", error);
    return { error: "Failed to delete object" };
  }
};
