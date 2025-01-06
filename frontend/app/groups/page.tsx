
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
import { revalidatePath, revalidateTag } from "next/cache";

const GroupsPage = async ({
  searchParams,
}: {
  searchParams: any;
}) => {

  console.log("params", searchParams);



  const { groups, total } = await getGroupsApi(searchParams);

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
      />

    </GenericLayout>
  );
};

export default GroupsPage;
