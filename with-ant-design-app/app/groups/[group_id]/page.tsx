"use client";

import React, { use, useEffect, useState } from "react";
import { Modal, Table } from "antd";
import { UserOutlined } from "@ant-design/icons";
import { useParams, useRouter } from "next/navigation";
import GenericLayout from "../../components/generic-page-layout";
import CardTable from "../../components/card-table";

import Layout from "react-masonry-list";
import MasonryTable from "../../components/masonry-table";
import {
  getObjects as getObjectsApi,
  createObject as createObjectApi,
  deleteObject as deleteObjectApi,
  updateObject as updateObjectApi,
} from "../../../actions/groupObjects";
import {
  getGroup as getGroupApi,
  getGroupsForSidebar as getGroupsForSidebarApi,
} from "../../../actions/group";
import Sidebar from "../../components/navigation/sidebar/Sidebar";

const GroupPage = ({
  searchParams,
}: {
  searchParams: any;
}) => {
  const router = useRouter();

  const params = useParams();

  const group_id = Array.isArray(params.group_id)
    ? params.group_id[0]
    : params.group_id;

    const [group, setGroup] = useState<any>(null);

  interface ObjectType {
    id: string;
    name: string;
    description: string;
    pictures: string[];
    groupId: string;
    createdAt: Date;
    updatedAt: Date;
  }

  const [objects, setObjects] = useState<ObjectType[]>([]);
  const [loading, setLoading] = useState(false);

  interface SidebarItemType {
    id: number;
    object_id: string;
    icon: JSX.Element;
    label: string;
    item: JSX.Element;
  }
  
  const [sidebarItems, setSidebarItems] = useState<SidebarItemType[]>([]);


  const fetchGroupDetails = async () => {
    try {
      console.log("group_id", group_id);
      const groupDetails = await getGroupApi(
        group_id as string
      );
      setGroup(groupDetails);
      console.log("groupDetails", groupDetails);
    } catch (error) {
      console.error("Error fetching group details:", error);
    }
  };

  const fetchObjects = async () => {
    setLoading(true);
    try {
      const objects = await getObjectsApi(group_id);
      if (Array.isArray(objects)) {
        console.log("objects", objects);
        setObjects(objects);
      } else {
        console.error(
          "Error fetching objects:",
          objects.error
        );
      }
    } catch (error) {
      console.error("Error fetching objects:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchGroupsForSidebar = async () => {
    try {
      const groups = await getGroupsForSidebarApi();
      
      if (Array.isArray(groups)) {
        setSidebarItems(
          groups.map((group, index) => ({
            id: index,
            object_id: group.id,
            icon: <UserOutlined />,
            label: group.name,
            item: <div>{group.name}</div>,
          }))
        );
      }

      console.log("groups", groups);
    } catch (error) {
      console.error("Error fetching groups:", error);
    }
  };

  useEffect(() => {
    fetchGroupsForSidebar();
    fetchObjects();
    fetchGroupDetails();
  }, []);

  const deleteObject = async (id: string) => {
    const result = await deleteObjectApi(id);
    await fetchObjects();
  };

  const updateObject = async (
    id: string,
    objectData: {
      name: string;
      description: string;
      pictures: string[];
    }
  ) => {
    const result = await updateObjectApi(id, objectData);
    await fetchObjects();
  };

  const openNewObjectForm = () => {
    router.push(`/groups/${group_id}/new-object`);
  };

  console.log("sidebarItems", sidebarItems);

  return (
    <GenericLayout>
      <div style={{ display: "flex", flexDirection: "row" }}>
      <Sidebar
        sidebarItems={sidebarItems}
        activeNavItem={sidebarItems ? sidebarItems.findIndex((item) => item.object_id === group_id) : 0}
        onNavClick={(object_id) => {router.push(`/groups/${object_id}`)}}
        title={group ? group.name : "Mana grupa"}
      />

      <MasonryTable
        columnCount={4}
        onCardEdit={(id) => {
          router.push(`/groups/${group_id}/${id}`);
        }}
        objects={objects}
        createObject={openNewObjectForm}
        deleteObject={deleteObject}
        updateObject={updateObject}
        loading={loading}
      />
      </div>
    </GenericLayout>
  );
};

export default GroupPage;
