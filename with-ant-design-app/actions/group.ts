"use server";

import { useSession } from "next-auth/react";
import { db } from "../db";
import { getServerSession } from "next-auth";
import { authOptions } from "../auth";
import { revalidatePath } from "next/cache";


// Get one group
export const getGroup = async (groupId: string) => {
  const session = await getServerSession(authOptions);
  // console.log("session in groups", session);
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
export const getGroups = async (filter: any) => {
  const session = await getServerSession(authOptions);
  console.log("session in groups", session);
  const user = session?.user;
  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }
  const userId = user.id;

  const filterGroupName = filter?.groupName || "";

  const page = Number.isInteger(parseInt(filter?.page)) ? parseInt(filter?.page) : 1;
  const pageSize = Number.isInteger(parseInt(filter?.pageSize)) ? parseInt(filter?.pageSize) : 6;
  const offset = (page - 1) * pageSize;

  // console.log("Filter:", filter);

  try {
    const groups = await db.residenceGroup.findMany({
      where: { 
      userId: userId,
      name: { contains: filterGroupName, mode: "insensitive" },
      },
      skip: offset,
      take: pageSize,
    });

    const total = await db.residenceGroup.count({
      where: { 
        userId: userId,
        name: { contains: filterGroupName, mode: "insensitive" },
      },
    });

    const groupsWithResidenceCount = await Promise.all(
      groups.map(async (group) => {
        const residenceCount = await db.residence.count({
          where: { groupId: group.id },
        });
        return { ...group, residenceCount };
      })
    );
    
    return { groups: groupsWithResidenceCount, total, error: null };
  } catch (error) { 
    console.error("Error getting groups:", error);
    return { groups: [], total: 0, error: "Failed to get groups" };
  }
};

// Get minimal group info for sidebar
export const getGroupsForSidebar = async () => {
  const session = await getServerSession(authOptions);
  // console.log("session in groups", session);
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

  console.log("groupName", groupName);

  const session = await getServerSession(authOptions);

  console.log("session in groups create", session);

  console.log("groupName", groupName);

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

    console.log("newGroup", newGroup);

    revalidatePath("/groups");

    return newGroup;


  } catch (error) {
    console.error("Error creating group:", error);
    return { error: "Failed to create group" };
  }
};

// Update an existing group
export const updateGroup = async (groupId: string, newGroupName: string) => {

  const session = await getServerSession(authOptions);
  
  const user = session?.user;

  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }

  // confirm that the user owns the group
  try {
    const group = await db.residenceGroup.findUnique({
      where: { id: groupId },
    });
    if (!group) {
      return { error: "Group not found or does not exist" };
    }
    if (group.userId !== user.id) {
      return { error: "Unauthorized" };
    }
  } catch (error) {
    console.error("Error updating group:", error);
    return { error: "Failed to update group" };
  }

  try {

    if (!newGroupName) {
      return { error: "Group name is required" };
    }
    // char count > 50 is not allowed
    if (newGroupName.length > 50) {
      return { error: "Group name must be 50 characters or less" };
    }

    const updatedGroup = await db.residenceGroup.update({
      where: { id: groupId },
      data: { 
        name: newGroupName, 
        updatedAt: new Date(),
      },
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
