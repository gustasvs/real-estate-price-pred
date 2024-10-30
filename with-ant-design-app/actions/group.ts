"use server";

import { useSession } from "next-auth/react";
import { db } from "../db";
import { auth } from "../auth";


// Get one group
export const getGroup = async (groupId: string) => {
  const session = await auth();
  console.log("session in groups", session);
  const user = session?.user;
  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }

  const userId = user.id;
  
  try {
    const group = await db.residenceGroup.findUnique({
      where: { id: groupId, userId: userId }, 
    });
    return group;
  } catch (error) {
    console.error("Error getting group:", error);
    return { error: "Failed to get group" };
  }
};

// Get all groups
export const getGroups = async () => {
const session = await auth();
  console.log("session in groups", session);
  const user = session?.user;
  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }
  const userId = user.id;

  try {
    const groups = await db.residenceGroup.findMany({
      where: { userId: userId },
    });
    return groups;
  } catch (error) { 
    console.error("Error getting groups:", error);
    return { error: "Failed to get groups" };
  }
};

// Get minimal group info for sidebar
export const getGroupsForSidebar = async () => {
  const session = await auth();
  console.log("session in groups", session);
  const user = session?.user;
  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }
  const userId = user.id;

  try {
    const groups = await db.residenceGroup.findMany({
      where: { userId: userId },
      select: {
        id: true, // include only necessary fields
        name: true,
        // add any other fields required for the sidebar
      },
    });
    return groups;
  } catch (error) { 
    console.error("Error getting groups for sidebar:", error);
    return { error: "Failed to get groups for sidebar" };
  }
};


// Create a new group
export const createGroup = async (groupName: string) => {

  const session = await auth();

  console.log("session in groups", session);

  const user = session?.user;

  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }

  const userId = user.id;
  
  try {
    const newGroup = await db.residenceGroup.create({
      data: {
        name: groupName,
        userId: userId,
      },
    });
    return newGroup;
  } catch (error) {
    console.error("Error creating group:", error);
    return { error: "Failed to create group" };
  }
};

// Update an existing group
export const updateGroup = async (groupId: string, newGroupName: string) => {
  try {
    const updatedGroup = await db.residenceGroup.update({
      where: { id: groupId },
      data: { name: newGroupName },
    });
    return updatedGroup;
  } catch (error) {
    console.error("Error updating group:", error);
    return { error: "Failed to update group" };
  }
};

// Delete an existing group
export const deleteGroup = async (groupId: string) => {
  try {
    await db.residenceGroup.delete({
      where: { id: groupId },
    });
    return { success: true };
  } catch (error) {
    console.error("Error deleting group:", error);
    return { error: "Failed to delete group" };
  }
};
