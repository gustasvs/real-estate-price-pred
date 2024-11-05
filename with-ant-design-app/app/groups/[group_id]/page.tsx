"use client";

import React, { use, useEffect, useState } from "react";
import { Modal, Table } from "antd";
import { UserOutlined } from "@ant-design/icons";
import { useParams, useRouter } from "next/navigation";
import GenericLayout from "../../components/generic-page-layout";
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
import PageHeader from "../../components/generic-page-layout/page-header/PageHeader";

export interface ResidenceObjectType {
  id: string;
  name?: string;
  address?: string;
  area?: number;
  description?: string;
  bedroomCount?: number;
  bathroomCount?: number;
  parkingCount?: number;
  price?: number;
  predictedPrice?: number;
  pictures?: { base64: string; status: string }[];
  favourite: boolean;
}

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

  const [objects, setObjects] = useState<
    ResidenceObjectType[]
  >([]);
  const [loading, setLoading] = useState(false);

  interface SidebarItemType {
    id: number;
    object_id: string;
    icon: JSX.Element;
    label: string;
    item: JSX.Element;
  }

  const [sidebarItems, setSidebarItems] = useState<
    SidebarItemType[]
  >([]);

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
        setObjects(
          objects.map((object) => ({
            ...object,
            pictures: object.pictures.map((picture) => ({
              base64: picture,
              status: "unknown",
            })),
          }))
        );
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
    objectData: ResidenceObjectType
  ) => {
    const result = await updateObjectApi(id, objectData);
    await fetchObjects();
  };

  const onCardFavorite = async (id: string) => {
    const object = objects.find(
      (object) => object.id === id
    );
    if (!object) {
      return;
    }
    const updatedObject = await updateObjectApi(id, {
      favourite: !object.favourite,
    });
    await fetchObjects();
  };

  const openNewObjectForm = () => {
    router.push(`/groups/${group_id}/new-object`);
  };

  console.log("sidebarItems", sidebarItems);

  return (
    <GenericLayout sidebarItems={sidebarItems}>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "2em",
        }}
      >
        {/* <Sidebar
        sidebarItems={sidebarItems}
        activeNavItem={sidebarItems ? sidebarItems.findIndex((item) => item.object_id === group_id) : 0}
        onNavClick={(object_id) => {router.push(`/groups/${object_id}`)}}
        title={group ? group.name : "Mana grupa"}
      /> */}
        <PageHeader
          title={group ? group.name : "Mana grupa"}
          breadcrumbItems={[
            { label: "Manas grupas", path: "/groups" },
            {
              label: group ? group.name : "Mana grupa",
              path: `/groups/${group_id}`,
            },
          ]}
        />
        <MasonryTable
          columnCount={4}
          objects={objects}
          onCardFavorite={onCardFavorite}
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
