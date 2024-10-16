"use client";

import React, { useEffect, useState } from "react";
import CardTable from "../components/card-table";
import GenericLayout from "../components/generic-page-layout";

import { useRouter } from "next/navigation";
import { Divider } from "antd";
import { PageHeader } from "@ant-design/pro-components";
import { createGroup  as createGroupApi, 
  deleteGroup as deleteGroupApi,
  getGroups as getGroupsApi,
  updateGroup as updateGroupApi} from "../../actions/group";
import { update } from "react-spring";

const GroupsPage = () => {

  const router = useRouter();

  const navigateToGroup = (id: number) => {
    console.log(`Open group with id ${id}`);
    router.push(`/groups/${id}`);
  };

  // const [groups, setGroups] = useState([
  //   { id: 1, name: "Rīgas dzīvokļi", imageUrl: "/path/to/image1.jpg" },
  //   { id: 2, name: "Cēsu dzīvokļi", imageUrl: "/path/to/image2.jpg" },
  //   { id: 3, name: "Mājas pie Jūrmalas", imageUrl: "/path/to/image3.jpg" },
  //   { id: 4, name: "Dzīvokļi S. Zvirbulim", imageUrl: "/path/to/image4.jpg" },
  //   // { id: 5, name: "Group 5", imageUrl: "/path/to/image5.jpg" },
  // ]);

  interface Group {
    id: string;
    name: string;
    imageUrl: any;
  }

  const [groups, setGroups] = useState<Group[]>([]);
  const [loading, setLoading] = useState(false);


  const getGroupsImage = (id: string) => {
    return `/path/to/image${id}.jpg`;
  };

  const fetchGroups = async () => {
    setLoading(true);
    try {
      const groups = await getGroupsApi();
      if (Array.isArray(groups)) {
        const groupsData = groups.map((group) => {
          return { id: group.id, name: group.name, imageUrl: getGroupsImage(group.id) };
        });
        console.log("groups", groupsData);
        setGroups(groupsData);
      } else {
        console.error("Error fetching groups:", groups.error);
      }
  } catch (error) {
    console.error("Error fetching groups:", error);
  } finally {
    setLoading(false);
  }
  };

  useEffect(() => {
    fetchGroups();
  }, []);


  const createGroup = async (groupName: string) => {

    const res = await createGroupApi(groupName);
    console.log("res", res);

    await fetchGroups();
  };

  const deleteGroup = async (id: string) => {
    const res = await deleteGroupApi(id);
    await fetchGroups();
  }

  const updateGroup = async (id: string, newGroupName: string) => {
    const res = await updateGroupApi(id, newGroupName);
    await fetchGroups();
  }


  return (
    <GenericLayout>
      <PageHeader
        ghost={false}
        onBack={() => window.history.back()}
        title="Manas grupas"
        subTitle="Šeit var redzēt visas manas grupas"
        // breadcrumb={{ routes }}
      >
      </PageHeader>
      <Divider />
        <CardTable columnCount={3} onCardClick={(id: number) => navigateToGroup(id)} groups={groups} deleteGroup={deleteGroup} 
        createGroup={createGroup} updateGroup={updateGroup} loading={loading}
        />
      {/* </div> */}
    </GenericLayout>
  );
};

export default GroupsPage;
