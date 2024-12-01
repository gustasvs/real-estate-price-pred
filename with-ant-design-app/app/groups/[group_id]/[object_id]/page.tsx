// "use client";

import { useParams, useRouter } from "next/navigation";
import GenericLayout from "../../../components/generic-page-layout";

import ResidenceObjectForm from "../../../components/masonry-table/ResidenceObjectForm";
import PageHeader from "../../../components/generic-page-layout/page-header/PageHeader";
import { useEffect, useState } from "react";
import { getGroup } from "../../../../actions/group";
import { getObject } from "../../../../actions/groupObjects";

const residenceObjectPage = async ({
  params,
}: {
  params: any;
}) => {
  console.log("params", params);

  const group_id = Array.isArray(params.group_id)
    ? params.group_id[0]
    : params.group_id;
  const object_id = Array.isArray(params.object_id)
    ? params.object_id[0]
    : params.object_id;

  const group = await getGroup(params.group_id);
  const residence = await getObject(params.object_id);

  const groupName =
    group && "name" in group ? group.name : "Mana grupa";
  const residenceName =
    residence && "name" in residence
      ? residence.name
      : "Mana dzīvesvieta";

  console.log("residence in page", residence);

  console.log("group", group);

  return (
    <GenericLayout>
      <PageHeader
        title={group ? groupName : "Mana grupa"}
        breadcrumbItems={[
          { label: "Manas grupas", path: "/groups" },
          {
            label: group ? groupName : "Mana grupa",
            path: `/groups/${group_id}`,
          },
          {
            label: residence
              ? residenceName
              : "Mana dzīvesvieta",
            path: `/groups/${group_id}/${object_id}`,
          },
        ]}
      />
      <ResidenceObjectForm
        objectId={object_id}
        groupId={group_id}
        residence={residence}
      />
    </GenericLayout>
  );
};

export default residenceObjectPage;
