"use client";

import React, { use, useEffect, useState } from "react";
import { Modal, Table } from "antd";
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
import { getGroup as getGroupApi } from "../../../actions/group";

export interface ResidenceObjectType {
  id: string;
  name?: string,
  address?: string,
  area?: number,
  description?: string,
  bedroomCount?: number,
  bathroomCount?: number,
  parkingCount?: number,
  price?: number,
  predictedPrice?: number,
  pictures?: string[]
}


const GroupPage = ({ searchParams }: { searchParams: any }) => {
  const router = useRouter();

  const params = useParams();
  
  const group_id = Array.isArray(params.group_id) ? params.group_id[0] : params.group_id;

  

  const [objects, setObjects] = useState<ResidenceObjectType[]>([]);
  const [loading, setLoading] = useState(false);

  const [groupDetails, setGroupDetails] = useState({});

  const fetchGroupDetails = async () => {
    try {
      const groupDetails = await getGroupApi(group_id as string);
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
        console.error("Error fetching objects:", objects.error);
      }
    } catch (error) {
      console.error("Error fetching objects:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
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

  const openNewObjectForm = () => {
    router.push(`/groups/${group_id}/new-object`);
  };

  return (
    <GenericLayout>
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
    </GenericLayout>
  );
};

export default GroupPage;
