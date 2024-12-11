import { revalidatePath } from "next/cache";
import { getMyFavoriteObjects } from "../../../actions/groupObjects";
import { generateDownloadUrl } from "../../api/generateDownloadUrl";
import GenericLayout from "../../components/generic-page-layout";
import PageHeader from "../../components/generic-page-layout/page-header/PageHeader";
import MasonryTable from "../../components/masonry-table";

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

const FavouriteObjectPage = async () => {

  const objectsResponse = await getMyFavoriteObjects();
  const objects = Array.isArray(objectsResponse) ? objectsResponse : [];

    console.log("objects", objects);

    // Add pre-signed download URLs to pictures
    const objectsWithPresignedDownloadUrls = await Promise.all(
      objects.map(async (obj) => {
        if (Array.isArray(obj.pictures)) {
          const updatedPictures = await Promise.all(
            obj.pictures.map(async (picture) => {
              const downloadUrl = await generateDownloadUrl(picture, 'object-pictures');
              return {
                fileName: picture,
                downloadUrl: typeof downloadUrl === 'object' && 'error' in downloadUrl ? null : downloadUrl,
              };
            })
          );
          return { ...obj, pictures: updatedPictures };
        }
        return obj;
      })
    );

  const revalidateData = async () => {
    "use server";
    revalidatePath(`/profile/favourites`);
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
          title={"Manas atzīmētās dzīvesvietas"}
          breadcrumbItems={[
            { label: "Mans profils", path: "/profile" },
            {
              label: "Manas atzīmētās dzīvesvietas",
              path: `/profile/favourites`,
            },
          ]}
        />
        <MasonryTable
          group_id={"favourites"}
          columnCount={4}
          objects={objectsWithPresignedDownloadUrls}
          loading={false}
          revalidateDataFunction={revalidateData}
        />
    </div>
    </GenericLayout>
  );
};

export default FavouriteObjectPage;
