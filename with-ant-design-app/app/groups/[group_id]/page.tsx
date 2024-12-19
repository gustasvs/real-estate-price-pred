import GenericLayout from "../../components/generic-page-layout";
import MasonryTable from "../../components/masonry-table";

import {
  getGroup as getGroupApi,
} from "../../../actions/group";
import PageHeader from "../../components/generic-page-layout/page-header/PageHeader";
import { getObjects } from "../../../actions/groupObjects";
import { revalidatePath } from "next/cache";
import { generateDownloadUrl } from "../../api/generateDownloadUrl";

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
  searchParams,
}: {
  params: any;
  searchParams: any;
}) => {

  console.log("params", params);

  if (!params.group_id) {
    return null;
  }

  const objectsResponse = await getObjects(params.group_id, searchParams);
  const objects = Array.isArray(objectsResponse) ? objectsResponse : [];

    // Add pre-signed download URLs to pictures
    // const objectsWithPresignedDownloadUrls = await Promise.all(
    //   objects.map(async (obj) => {
    //     if (Array.isArray(obj.pictures)) {
    //       const updatedPictures = await Promise.all(
    //         obj.pictures.map(async (picture) => {
    //           const downloadUrl = await generateDownloadUrl(picture, 'object-pictures');
    //           return {
    //             fileName: picture,
    //             downloadUrl: typeof downloadUrl === 'object' && 'error' in downloadUrl ? null : downloadUrl,
    //           };
    //         })
    //       );
    //       return { ...obj, pictures: updatedPictures };
    //     }
    //     return obj;
    //   })
    // );

    
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
          revalidateDataFunction={revalidateData}
        />
    </div>
    </GenericLayout>
  );
};

export default GroupPage;
