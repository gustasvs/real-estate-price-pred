"use server";

import { useSession } from "next-auth/react";
import { db } from "../db";
import { auth } from "../auth";
// import redisClient from "../redis/redisClient";

// async function addObjectToStream(objectId: string) {
//   await redisClient.xAdd('objectStream', '*', { objectId });
// }


export const getObject = async (objectId: string) => {
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

export const getObjects = async (groupId: string) => {
  try {
    const objects = await db.residence.findMany({
      where: { groupId: groupId },
    });
    return objects;
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
      // await manageObjectState(newObject.id, 'queued');

      // await redisClient.publish('objectCreationQueue', JSON.stringify({
      //   objectId: newObject.id,
      //   userId: user.id
      // }));
  
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
