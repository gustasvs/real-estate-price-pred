"use server";

import { useSession } from "next-auth/react";
import { db } from "../db";
import { auth } from "../auth";


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

  export const createObject = async (groupId: string, objectData: { 
    name: string, 
    address: string,
    area: number,
    description: string,
    bedroomCount: number,
    bathroomCount: number,
    parkingCount: number,
    price: number,
    predictedPrice: number,
    pictures: string[]
  }) => {
    try {

      console.log("objectData", objectData);

      const newObject = await db.residence.create({
        data: {
          name: objectData.name,
          address: objectData.address,
          description: objectData.description,
          pictures: objectData.pictures,
          groupId: groupId,
          area: objectData.area,
          bedroomCount: objectData.bedroomCount,
          bathroomCount: objectData.bathroomCount,
          parkingCount: objectData.parkingCount,
          price: objectData.price,
          predictedPrice: objectData.predictedPrice,
        },
      });
      return newObject;
    } catch (error) {
      console.error("Error creating object:", error);
      return { error: "Failed to create object" };
    }
  };


  export const updateObject = async (objectId: string, objectData: {
    name?: string,
    address?: string,
    area?: number,
    description?: string,
    bedroomCount?: number,
    bathroomCount?: number,
    parkingCount?: number,
    price?: number,
    predictedPrice?: number,
    pictures?: { base64: string, status: string }[]
  }) => {
    try {
      // Handle non-picture data updates separately
      const { pictures, ...dataWithoutPictures } = objectData;
  
      const updatedObject = await db.residence.update({
        where: { id: objectId },
        data: dataWithoutPictures,
      });
  
      // Only proceed with picture handling if pictures are provided
      if (pictures && Array.isArray(pictures)) {
        const filteredPictures = pictures.filter(p => p.status !== 'deleted').map(p => p.base64);
        const deletedPictures = pictures.filter(p => p.status === 'deleted').map(p => p.base64);
  
        // Compute new picture set by filtering out deleted ones and adding new ones
  const newPictures = updatedObject.pictures
  .filter(p => !deletedPictures.includes(p))
  .concat(filteredPictures);

await db.residence.update({
  where: { id: objectId },
  data: {
    pictures: newPictures,
  },
});

      }
  
      console.log("updatedObject", updatedObject);
      return updatedObject;
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
  