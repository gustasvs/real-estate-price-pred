"use server";

import { useSession } from "next-auth/react";
import { db } from "../db";
import { auth } from "../auth";


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

  
  export const createObject = async (groupId: string, objectData: { name: string, description: string, pictures: string[] }) => {
    try {

      console.log("objectData", objectData);

      const newObject = await db.residence.create({
        data: {
          name: objectData.name,
          description: objectData.description,
          pictures: objectData.pictures,
          groupId: groupId,
        },
      });
      return newObject;
    } catch (error) {
      console.error("Error creating object:", error);
      return { error: "Failed to create object" };
    }
  };


  export const updateObject = async (objectId: string, objectData: { name: string, description: string, pictures: string[] }) => {
    try {
      const updatedObject = await db.residence.update({
        where: { id: objectId },
        data: {
          name: objectData.name,
          description: objectData.description,
          pictures: objectData.pictures,
        },
      });
      return updatedObject;
    } catch (error) {
      console.error("Error updating object:", error);
      return { error: "Failed to update object" };
    }
  }

  

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
  