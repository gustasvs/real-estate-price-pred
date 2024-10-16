"use client";

import React, { useEffect, useState } from "react";
import { Modal, Table } from "antd";
import { useRouter } from "next/navigation";
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
import { getGroup as getGroupApi } from "../../../actions/group";

const GroupPage = ({ searchParams }: { searchParams: any }) => {
  const router = useRouter();
  const { group_id } = searchParams;

  // const [objects, setObjects] = useState([
  //   { id: 1, name: "Brīvības iela 12", imageUrl: "/path/to/image1.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   { id: 2, name: "Lāčplēša 45", imageUrl: "/path/to/image2.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   // { id: 3, name: "Valdemāra iela", imageUrl: "/path/to/image3.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   // { id: 4, name: "Pērnavas 6", imageUrl: "/path/to/image4.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   // { id: 5, name: "Krasta iela 24, Rīga", imageUrl: "/path/to/image5.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   // { id: 6, name: "Āgenskalns", imageUrl: "/path/to/image6.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   { id: 7, name: "Tērbatas iela 56", imageUrl: "/path/to/image7.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   // { id: 8, name: "Ventspils", imageUrl: "/path/to/image8.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   { id: 9, name: "Dzirnavu iela 112, Rīga", imageUrl: "/path/to/image9.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   // { id: 10, name: "Vecpilsēta", imageUrl: "/path/to/image10.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   { id: 11, name: "Salaspils 32", imageUrl: "/path/to/image11.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   // { id: 12, name: "Rīgas iela", imageUrl: "/path/to/image12.jpg", height: Math.floor(Math.random() * 200) + 100 },
  //   { id: 13, name: "Jelgavas 78", imageUrl: "/path/to/image13.jpg", height: Math.floor(Math.random() * 200) + 100 },
  // ]);
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

  const [groupDetails, setGroupDetails] = useState({});

  const fetchGroupDetails = async () => {
    try {
      const groupDetails = await getGroupApi(group_id);
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
        setObjects(objects);
      } else {
        console.error("Error fetching objects:", objects.error);
      }
    } catch (error) {
      console.error("Error fetching objects:", error);
    } finally {
      setLoading(false);
    }
  };

  const createObject = async (objectData: {
    name: string;
    description: string;
    pictures: string[];
  }) => {
    const newObject = await createObjectApi(group_id, objectData);

    await fetchObjects();
  };
  const deleteObject = async (id: string) => {
    const result = await deleteObjectApi(id);
    await fetchObjects();
  };
  const updateObject = async (
    id: string,
    objectData: { name: string; description: string; pictures: string[] }
  ) => {
    const result = await updateObjectApi(id, objectData);
    await fetchObjects();
  };

  const openNewObjectForm = () => {
    router.push(`/groups/${group_id}/new-object`);
  };

  return (
    <GenericLayout>
      <MasonryTable
        columnCount={4}
        onCardClick={(id) => {
          router.push(`/groups/${group_id}/objects/${id}`);
        }}
        objects={objects}
        createObject={openNewObjectForm}
        deleteObject={deleteObject}
        updateObject={updateObject}
        loading={loading}
      />
    </GenericLayout>
  );
};

export default GroupPage;
