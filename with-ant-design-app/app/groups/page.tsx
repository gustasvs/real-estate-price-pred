import React from "react";
import CardTable from "../components/card-table";
import GenericLayout from "../components/generic-page-layout";

import {
  createGroup as createGroupApi,
  deleteGroup as deleteGroupApi,
  getGroups as getGroupsApi,
  updateGroup as updateGroupApi,
} from "../../actions/group";
import PageHeader from "../components/generic-page-layout/page-header/PageHeader";
import { revalidatePath } from "next/cache";

const GroupsPage = async ({
  searchParams,
}: {
  searchParams: any;
}) => {

  console.log("params", searchParams);

  const fetchGroups = async () => {
    "use server";
    const { groups, total, error } = await getGroupsApi(
      searchParams
    );
    console.log("groups", groups);
    if (Array.isArray(groups)) {
      return { groups, total };
    } else {
      console.error(
        "Failed to fetch groups:",
        error
      );
      return { groups: [], total: 0 }
    }
  };

  const createGroup = async (groupName: string) => {
    "use server";
    const res = await createGroupApi(groupName);
    console.log("res", res);
    revalidatePath("/groups");
  };

  const deleteGroup = async (id: string) => {
    "use server";
    const res = await deleteGroupApi(id);
    revalidatePath("/groups");
  };

  const updateGroup = async (
    id: string,
    newGroupName: string
  ) => {
    "use server";
    const res = await updateGroupApi(id, newGroupName);
    revalidatePath("/groups");
  };

  const { groups, total } = await fetchGroups();

  return (
    <GenericLayout>
      <PageHeader
        title="Manas grupas"
        breadcrumbItems={[
          { label: "Manas grupas", path: "/groups" },
        ]}
      />

      <CardTable
        columnCount={3}
        groups={groups}
        total={total || 0}
        deleteGroup={deleteGroup}
        createGroup={createGroup}
        updateGroup={updateGroup}
      />

    </GenericLayout>
  );
};

export default GroupsPage;
