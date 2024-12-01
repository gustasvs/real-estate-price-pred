import { Modal, Table } from "antd";
import { UserOutlined } from "@ant-design/icons";
import GenericLayout from "../../components/generic-page-layout";
import MasonryTable from "../../components/masonry-table";

import {
  getGroup as getGroupApi,
  getGroupsForSidebar as getGroupsForSidebarApi,
} from "../../../actions/group";
import Sidebar from "../../components/navigation/sidebar/Sidebar";
import PageHeader from "../../components/generic-page-layout/page-header/PageHeader";
import { getObjects } from "../../../actions/groupObjects";
import { revalidatePath, revalidateTag } from "next/cache";

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

const GroupPage = async ({
  params,
}: {
  params: any;
}) => {

  console.log("params", params);

  if (!params.group_id) {
    return null;
  }

  const objectsResponse = await getObjects(params.group_id);
  const objects = Array.isArray(objectsResponse) ? objectsResponse : [];

  const groupResponse = await getGroupApi(params.group_id);
  const group = groupResponse ? groupResponse : null;

  const pageTitle = params.group_id !== "new" ? group && 'name' in group ? group.name : "Mana grupa" : "Jauna grupa";


  const group_id = Array.isArray(params.group_id)
    ? params.group_id[0]
    : params.group_id;

  console.log("group_id", group_id);

  const revalidateData = async () => {
    "use server";
    revalidatePath(`/groups/${group_id}`);
    // revalidateTag(group_id);
  }
  
  return (
    <GenericLayout >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "2em",
        }}
      >
         <PageHeader
          title={pageTitle}
          breadcrumbItems={[
            { label: "Manas grupas", path: "/groups" },
            {
              label: pageTitle,
              path: `/groups/${group_id}`,
            },
          ]}
        />
        <MasonryTable
          group_id={group_id}
          columnCount={4}
          objects={objects}
          loading={false}
          revalidateDataFunction={revalidateData}
        />
    </div>
    </GenericLayout>
  );
};

export default GroupPage;
