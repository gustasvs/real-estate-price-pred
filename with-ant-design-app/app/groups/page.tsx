// "use client";

import React, { useEffect, useState } from "react";
import CardTable from "../components/card-table";
import GenericLayout from "../components/generic-page-layout";

// import { useRouter } from "next/navigation";
import { Divider } from "antd";
import {
  createGroup as createGroupApi,
  deleteGroup as deleteGroupApi,
  getGroups as getGroupsApi,
  updateGroup as updateGroupApi,
} from "../../actions/group";
import { update } from "react-spring";
import PageHeader from "../components/generic-page-layout/page-header/PageHeader";
import { revalidatePath } from "next/cache";

const GroupsPage = async () => {
  interface Group {
    id: string;
    name: string;
    imageUrl: any;
  }

  const fetchGroups = async () => {
    "use server";
    const groups = await getGroupsApi();
    if (Array.isArray(groups)) {
      return groups;
    } else {
      console.error(
        "Failed to fetch groups:",
        groups.error
      );
      return [];
    }
  };

  const createGroup = async (groupName: string) => {
    "use server";
    const res = await createGroupApi(groupName);
    console.log("res", res);

    await fetchGroups();
  };

  const deleteGroup = async (id: string) => {
    "use server";
    const res = await deleteGroupApi(id);
    await fetchGroups();
  };

  const updateGroup = async (
    id: string,
    newGroupName: string
  ) => {
    "use server";
    const res = await updateGroupApi(id, newGroupName);
    revalidatePath("/groups");
  };

  const groups = await fetchGroups();

  return (
    <GenericLayout>
      <PageHeader
        title="Manas grupas"
        breadcrumbItems={[
          { label: "Manas grupas", path: "/groups" },
        ]}
      />

      <Divider />
      <CardTable
        columnCount={3}
        groups={groups}
        deleteGroup={deleteGroup}
        createGroup={createGroup}
        updateGroup={updateGroup}
      />
      {/* </div> */}
    </GenericLayout>
  );
};

export default GroupsPage;
